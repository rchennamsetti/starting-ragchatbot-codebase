// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');

    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // New chat
    newChatButton.addEventListener('click', startNewChat);

    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        const sourceHtml = renderGroupedSources(sources);

        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <ol class="sources-content">${sourceHtml}</ol>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Group flat sources by course, sorted alphabetically by course title and
// then numerically by lesson number, so each course lists its lessons as
// sub-items underneath it.
function groupSourcesByCourse(sources) {
    const courseMap = new Map();

    for (const source of sources) {
        if (!courseMap.has(source.course_title)) {
            courseMap.set(source.course_title, { courseLink: null, lessons: new Map() });
        }
        const entry = courseMap.get(source.course_title);

        if (source.lesson_number === null || source.lesson_number === undefined) {
            entry.courseLink = source.link;
        } else if (!entry.lessons.has(source.lesson_number)) {
            entry.lessons.set(source.lesson_number, source.link);
        }
    }

    const sortedCourseTitles = Array.from(courseMap.keys()).sort((a, b) => a.localeCompare(b));

    return sortedCourseTitles.map(courseTitle => {
        const { courseLink, lessons } = courseMap.get(courseTitle);
        const sortedLessonNumbers = Array.from(lessons.keys()).sort((a, b) => a - b);
        return {
            courseTitle,
            courseLink,
            lessons: sortedLessonNumbers.map(number => ({ number, link: lessons.get(number) }))
        };
    });
}

// Render a source label as a clickable (invisible-link-style) anchor when a
// URL is available, otherwise as plain text.
function renderSourceLabel(text, link) {
    const escapedText = escapeHtml(text);
    if (link) {
        return `<a href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer" class="source-link">${escapedText}</a>`;
    }
    return `<span class="source-link source-link--plain">${escapedText}</span>`;
}

function renderGroupedSources(sources) {
    const groupedCourses = groupSourcesByCourse(sources);

    return groupedCourses.map(course => {
        // A course with no lesson-level matches is clickable itself (via its
        // course link); otherwise it's just a plain group header.
        const courseLabel = course.lessons.length === 0
            ? renderSourceLabel(course.courseTitle, course.courseLink)
            : renderSourceLabel(course.courseTitle, null);

        let lessonListHtml = '';
        if (course.lessons.length > 0) {
            const lessonItems = course.lessons
                .map(lesson => `<li>${renderSourceLabel(`Lesson ${lesson.number}`, lesson.link)}</li>`)
                .join('');
            lessonListHtml = `<ol class="source-lessons">${lessonItems}</ol>`;
        }

        return `<li>${courseLabel}${lessonListHtml}</li>`;
    }).join('');
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Start a new chat: clear the current conversation and get a fresh session,
// tearing down the old session's history on the backend along the way.
async function startNewChat() {
    chatInput.disabled = true;
    sendButton.disabled = true;

    try {
        const response = await fetch(`${API_URL}/new-chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: currentSessionId })
        });

        if (!response.ok) throw new Error('Failed to start new chat');

        const data = await response.json();
        currentSessionId = data.session_id;
    } catch (error) {
        console.error('Error starting new chat:', error);
        // Fall back to a local reset - the next query will lazily create a session
        currentSessionId = null;
    } finally {
        chatMessages.innerHTML = '';
        addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}