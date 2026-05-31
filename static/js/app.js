/* ============================================
   IT Law Chatbot - Frontend Application
   ============================================ */

const API_BASE = '';  // Same origin
let currentConversationId = null;
let isLoading = false;
let currentChatAbortController = null; // Abort pending chat request on conversation switch
let conversationsCache = [];


// ---- DOM Elements ----
const chatArea = document.getElementById('chatArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');

const sidebar = document.getElementById('sidebar');

// ---- Theme Toggle ----
const btnThemeToggle = document.getElementById('btnThemeToggle');
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

btnThemeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});
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

    // Close sidebar button (inside sidebar)
    document.getElementById('btnCloseSidebar').addEventListener('click', toggleSidebar);

    // Search conversations
    document.getElementById('searchConversations').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        const items = conversationList.querySelectorAll('.conversation-item');
        items.forEach(item => {
            const title = item.querySelector('.conv-title')?.textContent.toLowerCase() || '';
            item.style.display = title.includes(query) ? '' : 'none';
        });
    });


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
async function apiCall(url, method = 'GET', data = null, signal = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (data) options.body = JSON.stringify(data);
    if (signal) options.signal = signal;

    const response = await fetch(`${API_BASE}${url}`, options);
    return response.json();
}

// ---- Conversations ----
async function loadConversations() {
    try {
        const result = await apiCall('/api/conversations');
        if (result.success) {
            conversationsCache = result.data;
            renderConversationList(conversationsCache);
        }
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

function renderConversationList(conversations) {
    // Lọc bỏ các cuộc hội thoại đã bị ẩn (Soft delete)
    const hiddenConvs = JSON.parse(localStorage.getItem('hiddenConversations') || '[]');
    const visibleConversations = conversations.filter(conv => !hiddenConvs.includes(conv.id));

    if (visibleConversations.length === 0) {
        conversationList.innerHTML = `
            <div class="conversation-empty">
                <p>Chưa có cuộc hội thoại nào</p>
            </div>`;
        return;
    }

    conversationList.innerHTML = visibleConversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}"
             onclick="loadConversation('${conv.id}')" data-id="${conv.id}">
            <div class="conv-title">${escapeHtml(conv.title)}</div>
            <div class="conv-meta">
                <span class="conv-time">${formatTime(conv.updated_at)}</span>
                <button class="btn-delete-conv" onclick="event.stopPropagation(); hideConversation('${conv.id}')" title="Xoá (chỉ ẩn khỏi màn hình)">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            </div>
        </div>
    `).join('');
}

function hideConversation(conversationId) {
    if (!confirm('Bạn có chắc muốn xoá cuộc hội thoại này? (Sẽ chỉ ẩn khỏi màn hình, dữ liệu vẫn an toàn)')) return;

    const hiddenConvs = JSON.parse(localStorage.getItem('hiddenConversations') || '[]');
    hiddenConvs.push(conversationId);
    localStorage.setItem('hiddenConversations', JSON.stringify(hiddenConvs));

    if (currentConversationId === conversationId) {
        newConversation();
    }
    loadConversations();
}

async function loadConversation(conversationId) {
    // Abort any pending chat request before switching
    if (currentChatAbortController) {
        currentChatAbortController.abort();
        currentChatAbortController = null;
        isLoading = false;
        btnSend.disabled = false;
    }

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
    // Abort any pending chat request before switching
    if (currentChatAbortController) {
        currentChatAbortController.abort();
        currentChatAbortController = null;
        isLoading = false;
        btnSend.disabled = false;
    }

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

    // Show typing indicator — kept alive until first text chunk from server
    const typingEl = showTypingIndicator();

    // Capture the conversation ID at send-time
    const sendConversationId = currentConversationId;

    // AbortController for fetch
    if (currentChatAbortController) currentChatAbortController.abort();
    currentChatAbortController = new AbortController();
    const signal = currentChatAbortController.signal;

    // --- Build assistant message container (hidden until first text) ---
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';
    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'message-body';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    bodyDiv.appendChild(contentDiv);
    msgDiv.appendChild(bodyDiv);

    let sources = null;
    let msgShown = false;

    function showMsgAndRemoveTyping() {
        if (!msgShown) {
            if (typingEl.parentNode) typingEl.remove();
            messagesContainer.appendChild(msgDiv);
            scrollToBottom();
            msgShown = true;
        }
    }

    // --- Typewriter queue ---
    // SSE pushes raw text into this queue; the animation loop drains it char-by-char
    let typeQueue = '';       // pending chars not yet displayed
    let displayedText = '';   // text already shown on screen
    let rawReceivedText = '';
    let queuedAnswerLength = 0;
    let thinkingEnded = false;
    let streamDone = false;   // SSE stream finished
    let typeAnimId = null;    // requestAnimationFrame id
    const CHAR_DELAY = 2;     // ms per character (adjust 5-50 for speed)
    let lastTypeTime = 0;

    function typewriterTick(ts) {
        if (!typeQueue.length && streamDone) {
            // All done — finalize
            return;
        }

        const elapsed = ts - lastTypeTime;
        const charsToAdd = Math.max(1, Math.floor(elapsed / CHAR_DELAY));

        if (typeQueue.length > 0 && elapsed >= CHAR_DELAY) {
            // If queue is falling behind (lots of pending chars), speed up
            const burst = typeQueue.length > 80 ? Math.min(typeQueue.length, 8) : charsToAdd;
            const chunk = typeQueue.slice(0, burst);
            typeQueue = typeQueue.slice(burst);
            displayedText += chunk;
            lastTypeTime = ts;

            // Re-render
            contentDiv.innerHTML = formatMessageContent(displayedText, true);
            scrollToBottom();
        }

        typeAnimId = requestAnimationFrame(typewriterTick);
    }

    // Start the animation loop
    typeAnimId = requestAnimationFrame((ts) => {
        lastTypeTime = ts;
        typeAnimId = requestAnimationFrame(typewriterTick);
    });

    // Helper: wait until typewriter drains the queue
    function waitForTypewriterDone() {
        return new Promise(resolve => {
            function check() {
                if (typeQueue.length === 0) { resolve(); return; }
                setTimeout(check, 20);
            }
            check();
        });
    }

    try {
        const response = await fetch(`${API_BASE}/api/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_id: currentConversationId,
            }),
            signal: signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (currentConversationId !== sendConversationId && sendConversationId !== null) {
            console.log('[sendMessage] Conversation switched, discarding response.');
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let sseBuffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            sseBuffer += decoder.decode(value, { stream: true });

            // Parse complete SSE frames (separated by \n\n)
            let boundary;
            while ((boundary = sseBuffer.indexOf('\n\n')) !== -1) {
                const frame = sseBuffer.slice(0, boundary);
                sseBuffer = sseBuffer.slice(boundary + 2);

                for (const line of frame.split('\n')) {
                    const clean = line.replace(/\r$/, '');
                    if (!clean.startsWith('data:')) continue;

                    const jsonStr = clean.slice(5).trim();
                    if (!jsonStr) continue;

                    let evt;
                    try { evt = JSON.parse(jsonStr); } catch (e) {
                        console.error('[SSE] JSON parse error:', e, jsonStr);
                        continue;
                    }

                    if (evt.event === 'metadata') {
                        const isNew = !currentConversationId;
                        currentConversationId = evt.conversation_id;
                        if (window.updateGraphVisualization && evt.graph_data) {
                            window.updateGraphVisualization(evt.graph_data);
                        }

                        // Local update to conversation sidebar list
                        if (isNew) {
                            const newConv = {
                                id: currentConversationId,
                                title: message.length > 50 ? message.substring(0, 47) + '...' : message,
                                created_at: new Date().toISOString(),
                                updated_at: new Date().toISOString()
                            };
                            conversationsCache = [newConv, ...conversationsCache];
                            renderConversationList(conversationsCache);
                        } else {
                            const idx = conversationsCache.findIndex(c => c.id === currentConversationId);
                            if (idx !== -1) {
                                const updated = { ...conversationsCache[idx], updated_at: new Date().toISOString() };
                                conversationsCache.splice(idx, 1);
                                conversationsCache = [updated, ...conversationsCache];
                                renderConversationList(conversationsCache);
                            }
                        }
                    } else if (evt.event === 'text') {
                        rawReceivedText += evt.text;
                        showMsgAndRemoveTyping();

                        const thinkingEndIdx = rawReceivedText.indexOf('</thinking>');
                        if (thinkingEndIdx === -1) {
                            if (rawReceivedText.includes('<thinking>') || '<thinking>'.startsWith(rawReceivedText.trim())) {
                                displayedText = rawReceivedText;
                                contentDiv.innerHTML = formatMessageContent(displayedText, true);
                                scrollToBottom();
                            } else {
                                const newChars = rawReceivedText.substring(queuedAnswerLength);
                                typeQueue += newChars;
                                queuedAnswerLength = rawReceivedText.length;
                            }
                        } else {
                            if (!thinkingEnded) {
                                thinkingEnded = true;
                                displayedText = rawReceivedText.substring(0, thinkingEndIdx + 11);
                                contentDiv.innerHTML = formatMessageContent(displayedText, true);
                                scrollToBottom();
                            }
                            const totalAnswer = rawReceivedText.substring(thinkingEndIdx + 11);
                            const newAnswerChars = totalAnswer.substring(queuedAnswerLength);
                            typeQueue += newAnswerChars;
                            queuedAnswerLength = totalAnswer.length;
                        }
                    } else if (evt.event === 'sources') {
                        sources = evt.sources;
                    } else if (evt.event === 'error') {
                        showMsgAndRemoveTyping();
                        cancelAnimationFrame(typeAnimId);
                        contentDiv.innerHTML = `❌ Lỗi: ${evt.error}`;
                    }
                }
            }
        }

        // SSE stream ended — wait for typewriter to finish rendering remaining queue
        streamDone = true;
        await waitForTypewriterDone();
        cancelAnimationFrame(typeAnimId);

        // Finalize: remove cursor, add copy button
        const copyBtnHtml = `
            <button class="btn-copy-msg" onclick="copyToClipboard(this)" title="Sao chép câu trả lời">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            </button>
        `;
        showMsgAndRemoveTyping();
        contentDiv.innerHTML = formatMessageContent(displayedText) + copyBtnHtml;

        // Render sources
        if (sources && sources.length > 0) {
            const sourcesHtml = `
                <div class="message-sources">
                    <div class="sources-title">Nguồn trích dẫn</div>
                    ${sources.map((s) => {
                const fullContent = escapeHtml(s.full_content || s.content || '');
                const docTitle = escapeHtml(s.doc_title || '');
                const soHieu = escapeHtml(s.so_hieu || '');
                const article = escapeHtml(s.article || '');
                return `
                        <div class="source-item source-item-clickable"
                             onclick="openSourceModal(this)"
                             data-full-content="${fullContent}"
                             data-doc-title="${docTitle}"
                             data-so-hieu="${soHieu}"
                             data-article="${article}">
                            <span class="source-item-text">${docTitle} ${soHieu ? '(' + soHieu + ')' : ''} ${article ? '- ' + article : ''}</span>
                            <svg class="source-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </div>`;
            }).join('')}
                </div>`;
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = sourcesHtml;
            bodyDiv.appendChild(tempDiv.firstElementChild);
        }

    } catch (e) {
        cancelAnimationFrame(typeAnimId);
        if (e.name === 'AbortError') {
            console.log('[sendMessage] Request aborted.');
            return;
        }
        if (typingEl.parentNode) typingEl.remove();
        appendMessage('assistant', '❌ Lỗi kết nối server. Vui lòng kiểm tra server đang chạy.');
        console.error('Send message error:', e);
    } finally {
        currentChatAbortController = null;
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
                    ${parsedSources.map((s, i) => {
                const fullContent = escapeHtml(s.full_content || s.content || '');
                const docTitle = escapeHtml(s.doc_title || '');
                const soHieu = escapeHtml(s.so_hieu || '');
                const article = escapeHtml(s.article || '');
                return `
                        <div class="source-item source-item-clickable"
                             onclick="openSourceModal(this)"
                             data-full-content="${fullContent}"
                             data-doc-title="${docTitle}"
                             data-so-hieu="${soHieu}"
                             data-article="${article}">
                            <span class="source-item-text">${docTitle} ${soHieu ? '(' + soHieu + ')' : ''} ${article ? '- ' + article : ''}</span>
                            <svg class="source-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </div>`;
            }).join('')}
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

async function typeWriterHTML(el, htmlString, speed = 15) {
    el.innerHTML = '';

    // Adaptive speed: short text (CHATCHIT) → slower, visible effect
    // Long text (LUAT) → faster, skip more chars per frame
    const plainTextLen = htmlString.replace(/<[^>]*>/g, '').length;
    const charsPerFrame = plainTextLen < 150 ? 3 : 20;
    const frameDelay = plainTextLen < 150 ? 20 : speed;

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
            cursor += charsPerFrame;
            if (cursor > htmlString.length) cursor = htmlString.length;
        }

        el.innerHTML = htmlString.substring(0, cursor);
        scrollToBottom();
        await new Promise(r => setTimeout(r, frameDelay));
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

function formatMessageContent(content, isStreaming = false) {
    if (!content) return '';

    let thinkingContent = '';
    let answerContent = '';
    let isThinking = false;
    let isAnswering = false;

    const thinkingStartIdx = content.indexOf('<thinking>');
    const thinkingEndIdx = content.indexOf('</thinking>');
    const answerStartIdx = content.indexOf('<answer>');
    const answerEndIdx = content.indexOf('</answer>');

    if (thinkingStartIdx !== -1) {
        if (thinkingEndIdx !== -1) {
            thinkingContent = content.substring(thinkingStartIdx + 10, thinkingEndIdx).trim();
            if (answerStartIdx !== -1) {
                if (answerEndIdx !== -1) {
                    answerContent = content.substring(answerStartIdx + 8, answerEndIdx).trim();
                } else {
                    answerContent = content.substring(answerStartIdx + 8).trim();
                    isAnswering = true;
                }
            } else {
                const remaining = content.substring(thinkingEndIdx + 11).trim();
                if (remaining.startsWith('<')) {
                    isAnswering = true;
                } else if (remaining) {
                    answerContent = remaining;
                    isAnswering = true;
                }
            }
        } else {
            thinkingContent = content.substring(thinkingStartIdx + 10).trim();
            isThinking = true;
        }
    } else {
        const cleanContent = content.trim();
        if (cleanContent.startsWith('<') && !cleanContent.includes('>') && '<thinking>'.startsWith(cleanContent)) {
            isThinking = true;
        } else {
            answerContent = content;
            isAnswering = isStreaming;
        }
    }

    const cleanTrailingTag = (text) => {
        return text.replace(/<[a-zA-Z\/]*$/g, '');
    };

    thinkingContent = cleanTrailingTag(thinkingContent);
    answerContent = cleanTrailingTag(answerContent);

    let thinkingHtml = '';
    if (thinkingContent || isThinking) {
        let formattedThinking = escapeHtml(thinkingContent);
        formattedThinking = formattedThinking.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedThinking = formattedThinking.replace(/\n/g, '<br>');

        thinkingHtml = `
            <details class="thinking-block" ${isThinking ? 'open' : ''}>
                <summary>Quá trình AI suy luận (Nhấp để mở)</summary>
                <div class="thinking-content">${formattedThinking}</div>
            </details>
        `;
    }

    let html = escapeHtml(answerContent);

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

// ---- Source Modal ----
function openSourceModal(el) {
    const modal = document.getElementById('sourceModal');
    const title = document.getElementById('sourceModalTitle');
    const subtitle = document.getElementById('sourceModalSubtitle');
    const body = document.getElementById('sourceModalBody');

    const article = el.dataset.article || '';
    const docTitle = el.dataset.docTitle || '';
    const soHieu = el.dataset.soHieu || '';

    title.textContent = article || docTitle;
    subtitle.textContent = docTitle + (soHieu ? ` (${soHieu})` : '');

    // Show loading state
    body.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px">Đang tải nội dung điều luật...</div>';
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Extract dieu_so from article string: "Điều 83. ..." → "83"
    const dieuMatch = article.match(/Điều\s+(\d+)/);
    const dieuSo = dieuMatch ? dieuMatch[1] : '';

    if (!dieuSo) {
        // Fallback to data attribute content
        const fallback = el.dataset.fullContent || 'Không có nội dung chi tiết.';
        body.innerHTML = `<div class="source-legal-text">${formatLegalText(fallback)}</div>`;
        return;
    }

    // Fetch full article from API
    fetch(`/api/article-full?dieu_so=${encodeURIComponent(dieuSo)}&so_hieu=${encodeURIComponent(soHieu)}`)
        .then(r => r.json())
        .then(result => {
            if (result.success && result.data) {
                const d = result.data;
                // Update title with full info from API
                if (d.dieu_ten) {
                    title.textContent = `Điều ${d.dieu_so}. ${d.dieu_ten}`;
                }
                if (d.ten_van_ban) {
                    subtitle.textContent = d.ten_van_ban + (d.so_hieu ? ` (${d.so_hieu})` : '');
                    if (d.chuong_ten) {
                        subtitle.textContent += ` — Chương ${d.chuong_so}: ${d.chuong_ten}`;
                    }
                }
                body.innerHTML = `<div class="source-legal-text">${formatLegalText(d.full_text)}</div>`;
            } else {
                body.innerHTML = '<div style="color:var(--danger);padding:20px">Không tìm thấy nội dung.</div>';
            }
        })
        .catch(() => {
            // Fallback to data attribute
            const fallback = el.dataset.fullContent || 'Lỗi tải nội dung.';
            body.innerHTML = `<div class="source-legal-text">${formatLegalText(fallback)}</div>`;
        });
}

function formatLegalText(text) {
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>')
        .replace(/(Điều\s+\d+[a-zđ]*\.?\s*[^<]*)/gi, '<strong>$1</strong>')
        .replace(/(Khoản\s+\d+)/gi, '<strong>$1</strong>')
        .replace(/(Điểm\s+[a-zđ]\))/gi, '<strong>$1</strong>');
}

function closeSourceModal() {
    const modal = document.getElementById('sourceModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

document.getElementById('sourceModalClose').addEventListener('click', closeSourceModal);
document.getElementById('sourceModal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeSourceModal();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSourceModal();
});
