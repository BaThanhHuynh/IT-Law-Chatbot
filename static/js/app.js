/* ============================================
   IT Law Chatbot - Frontend Application
   ============================================ */

const API_BASE = '';  // Same origin
let currentConversationId = null;
let isLoading = false;


// ---- DOM Elements ----
const chatArea = document.getElementById('chatArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');

const sidebar = document.getElementById('sidebar');

const conversationList = document.getElementById('conversationList');
const chatTitle = document.getElementById('chatTitle');

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    setupEventListeners();
});

function setupEventListeners() {
    btnSend.addEventListener('click', sendMessage);
    btnNewChat.addEventListener('click', newConversation);
    btnToggleSidebar.addEventListener('click', toggleSidebar);


    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    });
}

// ---- API Calls ----
async function apiCall(url, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (data) options.body = JSON.stringify(data);

    const response = await fetch(`${API_BASE}${url}`, options);
    return response.json();
}

// ---- Conversations ----
async function loadConversations() {
    try {
        const result = await apiCall('/api/conversations');
        if (result.success) {
            renderConversationList(result.data);
        }
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

function renderConversationList(conversations) {
    if (!conversations || conversations.length === 0) {
        conversationList.innerHTML = `
            <div class="conversation-empty">
                <p>Chưa có cuộc hội thoại nào</p>
            </div>`;
        return;
    }

    conversationList.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}"
             onclick="loadConversation('${conv.id}')" data-id="${conv.id}">
            <div class="conv-title">${escapeHtml(conv.title)}</div>
            <div class="conv-time">${formatTime(conv.updated_at)}</div>
        </div>
    `).join('');
}

async function loadConversation(conversationId) {
    currentConversationId = conversationId;

    // Update UI
    welcomeScreen.style.display = 'none';
    messagesContainer.style.display = 'flex';
    messagesContainer.innerHTML = '';

    // Update active state in sidebar
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === conversationId);
    });

    try {
        const result = await apiCall(`/api/conversations/${conversationId}`);
        if (result.success) {
            result.data.forEach(msg => {
                appendMessage(msg.role, msg.content, msg.sources);
            });
            scrollToBottom();
        }
    } catch (e) {
        console.error('Failed to load conversation:', e);
    }

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('show');
    }
}

function newConversation() {
    currentConversationId = null;
    welcomeScreen.style.display = 'flex';
    messagesContainer.style.display = 'none';
    messagesContainer.innerHTML = '';
    chatTitle.textContent = 'ITL Assistant';

    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });

    messageInput.focus();
}

// ---- Messages ----
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    btnSend.disabled = true;
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show message area
    welcomeScreen.style.display = 'none';
    messagesContainer.style.display = 'flex';

    // Append user message
    appendMessage('user', message);
    scrollToBottom();

    // Show typing indicator
    const typingEl = showTypingIndicator();

    try {
        const result = await apiCall('/api/chat', 'POST', {
            message: message,
            conversation_id: currentConversationId,
        });

        // Remove typing indicator
        typingEl.remove();

        if (result.success) {
            const data = result.data;
            currentConversationId = data.conversation_id;

            // Append assistant message with animation (animate = true)
            appendMessage('assistant', data.answer, data.sources, true);


            // Refresh conversation list
            loadConversations();
        } else {
            appendMessage('assistant', `❌ Lỗi: ${result.error || 'Không xác định'}`);
        }
    } catch (e) {
        typingEl.remove();
        appendMessage('assistant', '❌ Lỗi kết nối server. Vui lòng kiểm tra server đang chạy.');
        console.error('Send message error:', e);
    }

    isLoading = false;
    btnSend.disabled = false;
    scrollToBottom();
    messageInput.focus();
}

function sendSuggestion(text) {
    messageInput.value = text;
    sendMessage();
}

function appendMessage(role, content, sources = null, animate = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const formattedContent = formatMessageContent(content);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const parsedSources = typeof sources === 'string' ? JSON.parse(sources) : sources;
        if (parsedSources && parsedSources.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <div class="sources-title">Nguồn trích dẫn</div>
                    ${parsedSources.map(s => `
                        <div class="source-item">
                            <span>${escapeHtml(s.doc_title || '')} ${s.so_hieu ? '(' + escapeHtml(s.so_hieu) + ')' : ''} ${s.article ? '- ' + escapeHtml(s.article) : ''}</span>
                            <span class="source-score">${(s.score * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>`;
        }
    }

    const copyBtnHtml = role === 'assistant' ? `
        <button class="btn-copy-msg" onclick="copyToClipboard(this)" title="Sao chép câu trả lời">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        </button>
    ` : '';

    msgDiv.innerHTML = `
        <div class="message-body">
            <div class="message-content">
                ${animate ? '' : formattedContent + copyBtnHtml}
            </div>
            ${animate ? '' : sourcesHtml}
        </div>
    `;

    messagesContainer.appendChild(msgDiv);

    if (animate) {
        const contentEl = msgDiv.querySelector('.message-content');
        const bodyEl = msgDiv.querySelector('.message-body');

        typeWriterHTML(contentEl, formattedContent, 15).then(() => {
            contentEl.innerHTML = formattedContent + copyBtnHtml;
            if (sourcesHtml) {
                // Reveal sources after typing is done
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = sourcesHtml;
                bodyEl.appendChild(tempDiv.firstElementChild);
            }
            scrollToBottom();
        });
    }
}

async function typeWriterHTML(el, htmlString, speed = 10) {
    el.innerHTML = '';
    let cursor = 0;
    while (cursor < htmlString.length) {
        if (htmlString[cursor] === '<') {
            let endTag = htmlString.indexOf('>', cursor);
            if (endTag !== -1) {
                cursor = endTag + 1;
            } else {
                cursor++;
            }
        } else if (htmlString[cursor] === '&') {
            let endEntity = htmlString.indexOf(';', cursor);
            if (endEntity !== -1 && endEntity - cursor < 10) {
                cursor = endEntity + 1;
            } else {
                cursor++;
            }
        } else {
            cursor += 20; // Tăng lên 20 ký tự mỗi lần
            if (cursor > htmlString.length) cursor = htmlString.length;
        }

        // Thêm con trỏ nhấp nháy giả
        el.innerHTML = htmlString.substring(0, cursor) + '<span style="border-right: 2px solid var(--text-color); margin-left: 2px; animation: blink 1s step-end infinite;"></span>';
        scrollToBottom();
        await new Promise(r => setTimeout(r, speed));
    }
    el.innerHTML = htmlString;
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-body">
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
}

function formatMessageContent(content) {
    if (!content) return '';

    let thinkingHtml = '';
    const thinkingMatch = content.match(/<thinking>([\s\S]*?)<\/thinking>/);
    if (thinkingMatch) {
        let thinkingContent = escapeHtml(thinkingMatch[1].trim());
        // Simple markdown for thinking block
        thinkingContent = thinkingContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        thinkingContent = thinkingContent.replace(/\n/g, '<br>');

        thinkingHtml = `
            <details class="thinking-block">
                <summary>Quá trình AI suy luận (Nhấp để mở)</summary>
                <div class="thinking-content">${thinkingContent}</div>
            </details>
        `;
        content = content.replace(/<thinking>[\s\S]*?<\/thinking>/, '');
    }

    // Extract <answer> block if any
    const answerMatch = content.match(/<answer>([\s\S]*?)<\/answer>/);
    if (answerMatch) {
        content = answerMatch[1].trim();
    }

    // Basic markdown-like formatting
    let html = escapeHtml(content);

    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text*
    html = html.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Lists: - item or * item
    html = html.replace(/^[\-\*]\s+(.+)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Numbered lists
    html = html.replace(/^\d+\.\s+(.+)/gm, '<li>$1</li>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return thinkingHtml + html;
}



// ---- Sidebar Toggle ----
function toggleSidebar() {
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('show');
    } else {
        sidebar.classList.toggle('hidden');
    }
}

// ---- Utilities ----
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'Vừa xong';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} phút trước`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} giờ trước`;
    return date.toLocaleDateString('vi-VN');
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatArea.scrollTo({
            top: chatArea.scrollHeight,
            behavior: 'smooth'
        });
    });
}

function copyToClipboard(btn) {
    const content = btn.closest('.message-content').innerText;
    navigator.clipboard.writeText(content).then(() => {
        const originalHtml = btn.innerHTML;
        btn.innerHTML = '<span style="font-size: 10px; color: var(--success)">Đã chép!</span>';
        setTimeout(() => {
            btn.innerHTML = originalHtml;
        }, 2000);
    });
}
