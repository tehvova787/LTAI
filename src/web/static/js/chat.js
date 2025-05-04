/**
 * Lucky Train AI Assistant - Chat functionality
 * 
 * This script handles the chat functionality for the web interface
 * including sending messages, receiving responses, and updating the UI.
 */

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    
    let sessionId = localStorage.getItem('session_id') || '';
    
    // Initialize autosize for textarea
    initializeAutosize();
    
    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to the chat
        addMessage('user', message);
        
        // Clear input
        messageInput.value = '';
        
        // Reset textarea height
        messageInput.style.height = 'auto';
        
        // Send message to API
        sendMessage(message);
    });
    
    // Clear chat button
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear the chat history?')) {
                clearChat();
            }
        });
    }
    
    // Theme toggle button
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function() {
            toggleTheme();
        });
    }
    
    // Load chat history from localStorage
    loadChatHistory();
    
    // Auto-resize textarea as user types
    function initializeAutosize() {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Add a message to the chat
    function addMessage(role, content, messageId) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        if (messageId) {
            messageDiv.dataset.messageId = messageId;
        }
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const avatarImg = document.createElement('img');
        avatarImg.src = role === 'user' 
            ? '/static/images/user-avatar.png' 
            : '/static/images/assistant-avatar.png';
        avatarImg.alt = role === 'user' ? 'User' : 'Assistant';
        
        avatarDiv.appendChild(avatarImg);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Add the message content
        if (typeof content === 'string') {
            // Render markdown-like format
            const formattedContent = formatMarkdown(content);
            contentDiv.innerHTML = formattedContent;
        } else {
            contentDiv.textContent = 'Error: Invalid message content';
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        // Add feedback buttons for assistant messages
        if (role === 'assistant' && messageId) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'message-feedback';
            feedbackDiv.innerHTML = `
                <button class="feedback-btn" data-rating="positive" title="This was helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                </button>
                <button class="feedback-btn" data-rating="negative" title="This was not helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
                </button>
            `;
            
            // Add event listeners for feedback buttons
            feedbackDiv.querySelectorAll('.feedback-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const rating = this.dataset.rating;
                    sendFeedback(messageId, rating);
                    
                    // Show feedback confirmation
                    feedbackDiv.innerHTML = '<span class="feedback-thanks">Thank you for your feedback!</span>';
                });
            });
            
            contentDiv.appendChild(feedbackDiv);
        }
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Save to chat history
        saveMessage(role, content, messageId);
    }
    
    // Format markdown-like text
    function formatMarkdown(text) {
        // Handle code blocks
        text = text.replace(/```(.*?)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        
        // Handle inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Handle bold
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Handle italic
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Handle links
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Handle lists
        text = text.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
        
        // Handle line breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    // Send message to API
    function sendMessage(message) {
        // Show typing indicator
        showTypingIndicator();
        
        // Check if streaming is enabled
        const useStreaming = true; // Set to false to disable streaming
        
        if (useStreaming) {
            sendStreamingMessage(message);
        } else {
            sendRegularMessage(message);
        }
    }
    
    // Send message using the regular (non-streaming) API
    function sendRegularMessage(message) {
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Update session ID
            if (data.session_id) {
                sessionId = data.session_id;
                localStorage.setItem('session_id', sessionId);
            }
            
            // Add response to chat
            if (data.response) {
                addMessage('assistant', data.response, data.message_id);
            } else if (data.error) {
                addErrorMessage(data.error);
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            addErrorMessage('Failed to send message. Please try again.');
        });
    }
    
    // Send message using the streaming API
    function sendStreamingMessage(message) {
        const eventSource = new EventSource(`/api/stream-chat?message=${encodeURIComponent(message)}&session_id=${sessionId}`);
        
        let messageId = null;
        let fullResponse = '';
        let responseElement = null;
        
        eventSource.onopen = function() {
            // The connection was opened
            console.log('SSE connection opened');
            
            // Create an empty response message
            responseElement = document.createElement('div');
            responseElement.className = 'message assistant';
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'message-avatar';
            
            const avatarImg = document.createElement('img');
            avatarImg.src = '/static/images/assistant-avatar.png';
            avatarImg.alt = 'Assistant';
            
            avatarDiv.appendChild(avatarImg);
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = '<p></p>';
            
            responseElement.appendChild(avatarDiv);
            responseElement.appendChild(contentDiv);
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add the empty message
            chatMessages.appendChild(responseElement);
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    console.error('Error in stream:', data.error);
                    eventSource.close();
                    
                    // Remove the partial response if any
                    if (responseElement) {
                        responseElement.remove();
                    }
                    
                    addErrorMessage(data.error);
                    return;
                }
                
                if (data.content) {
                    fullResponse += data.content;
                    
                    // Update the response message
                    const contentElement = responseElement.querySelector('.message-content p');
                    contentElement.innerHTML = formatMarkdown(fullResponse);
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                if (data.end) {
                    eventSource.close();
                    
                    // Update session ID
                    if (data.session_id) {
                        sessionId = data.session_id;
                        localStorage.setItem('session_id', sessionId);
                    }
                    
                    // Save the message ID
                    messageId = data.message_id;
                    responseElement.dataset.messageId = messageId;
                    
                    // Add feedback buttons
                    if (data.feedback && messageId) {
                        const contentDiv = responseElement.querySelector('.message-content');
                        
                        const feedbackDiv = document.createElement('div');
                        feedbackDiv.className = 'message-feedback';
                        feedbackDiv.innerHTML = `
                            <button class="feedback-btn" data-rating="positive" title="This was helpful">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                            </button>
                            <button class="feedback-btn" data-rating="negative" title="This was not helpful">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
                            </button>
                        `;
                        
                        // Add event listeners for feedback buttons
                        feedbackDiv.querySelectorAll('.feedback-btn').forEach(btn => {
                            btn.addEventListener('click', function() {
                                const rating = this.dataset.rating;
                                sendFeedback(messageId, rating);
                                
                                // Show feedback confirmation
                                feedbackDiv.innerHTML = '<span class="feedback-thanks">Thank you for your feedback!</span>';
                            });
                        });
                        
                        contentDiv.appendChild(feedbackDiv);
                    }
                    
                    // Save the message
                    saveMessage('assistant', fullResponse, messageId);
                }
            } catch (error) {
                console.error('Error parsing SSE message:', error);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE error:', error);
            eventSource.close();
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Remove the partial response if any
            if (responseElement) {
                responseElement.remove();
            }
            
            addErrorMessage('Connection error. Please try again.');
        };
    }
    
    // Add an error message to the chat
    function addErrorMessage(errorText) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system error';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<p>Error: ${errorText}</p>`;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Show typing indicator
    function showTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'message assistant typing-indicator';
        indicatorDiv.id = 'typing-indicator';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const avatarImg = document.createElement('img');
        avatarImg.src = '/static/images/assistant-avatar.png';
        avatarImg.alt = 'Assistant';
        
        avatarDiv.appendChild(avatarImg);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        
        indicatorDiv.appendChild(avatarDiv);
        indicatorDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(indicatorDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Send feedback
    function sendFeedback(messageId, rating) {
        fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating === 'positive' ? 1 : 0
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback sent:', data);
        })
        .catch(error => {
            console.error('Error sending feedback:', error);
        });
    }
    
    // Save a message to the chat history
    function saveMessage(role, content, messageId) {
        const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
        
        history.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            messageId: messageId
        });
        
        // Limit history length
        const maxHistory = 100;
        if (history.length > maxHistory) {
            history.splice(0, history.length - maxHistory);
        }
        
        localStorage.setItem('chat_history', JSON.stringify(history));
    }
    
    // Load chat history from localStorage
    function loadChatHistory() {
        const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
        
        if (history.length > 0) {
            // Clear the initial assistant message
            chatMessages.innerHTML = '';
            
            // Add messages from history
            history.forEach(msg => {
                addMessage(msg.role, msg.content, msg.messageId);
            });
        }
    }
    
    // Clear chat history
    function clearChat() {
        // Clear UI
        chatMessages.innerHTML = '';
        
        // Add initial message
        addMessage('assistant', 'Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?');
        
        // Clear localStorage
        localStorage.removeItem('chat_history');
        
        // Optionally reset session
        sessionId = '';
        localStorage.removeItem('session_id');
    }
    
    // Toggle between light and dark theme
    function toggleTheme() {
        const body = document.body;
        const isDark = body.classList.contains('theme-dark');
        
        if (isDark) {
            body.classList.remove('theme-dark');
            body.classList.add('theme-light');
            localStorage.setItem('theme', 'light');
        } else {
            body.classList.remove('theme-light');
            body.classList.add('theme-dark');
            localStorage.setItem('theme', 'dark');
        }
    }
    
    // Load saved theme
    function loadSavedTheme() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.body.classList.remove('theme-light', 'theme-dark');
            document.body.classList.add(`theme-${savedTheme}`);
        }
    }
    
    // Load saved theme
    loadSavedTheme();
});

// Add a CSS class to make links with target="_blank" secure
document.addEventListener('DOMContentLoaded', function() {
    const externalLinks = document.querySelectorAll('a[target="_blank"]');
    
    externalLinks.forEach(link => {
        if (!link.hasAttribute('rel')) {
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
}); 