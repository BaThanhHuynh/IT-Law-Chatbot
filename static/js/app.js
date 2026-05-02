/* ============================================
   IT Law Chatbot - Frontend Application
   ============================================ */

const API_BASE = '';  // Same origin
let currentConversationId = null;
let isLoading = false;
let graphVisible = false;
let graphData = { nodes: [], edges: [] };

// ---- DOM Elements ----
const chatArea = document.getElementById('chatArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');
const btnToggleGraph = document.getElementById('btnToggleGraph');
const btnCloseGraph = document.getElementById('btnCloseGraph');
const sidebar = document.getElementById('sidebar');
const graphPanel = document.getElementById('graphPanel');
const graphCanvas = document.getElementById('graphCanvas');
const conversationList = document.getElementById('conversationList');
const chatTitle = document.getElementById('chatTitle');

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    setupEventListeners();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
});

function setupEventListeners() {
    btnSend.addEventListener('click', sendMessage);
    btnNewChat.addEventListener('click', newConversation);
    btnToggleSidebar.addEventListener('click', toggleSidebar);
    btnToggleGraph.addEventListener('click', toggleGraph);
    btnCloseGraph.addEventListener('click', toggleGraph);

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
    chatTitle.textContent = 'Tư vấn Luật Công nghệ thông tin';

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

            // Append assistant message
            appendMessage('assistant', data.answer, data.sources);

            // Update graph
            if (data.graph_data && (data.graph_data.nodes.length > 0)) {
                graphData = data.graph_data;
                drawGraph();
            }

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

function appendMessage(role, content, sources = null) {
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
                ${formattedContent}
                ${copyBtnHtml}
            </div>
            ${sourcesHtml}
        </div>
    `;

    messagesContainer.appendChild(msgDiv);
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

    return html;
}

// ---- Knowledge Graph Visualization ----
function toggleGraph() {
    graphVisible = !graphVisible;
    graphPanel.classList.toggle('visible', graphVisible);
    btnToggleGraph.classList.toggle('active', graphVisible);

    if (graphVisible) {
        setTimeout(() => {
            resizeCanvas();
            if (graphData.nodes.length > 0) {
                drawGraph();
            } else {
                loadFullGraph();
            }
        }, 300);
    }
}

async function loadFullGraph() {
    try {
        const result = await apiCall('/api/knowledge-graph?depth=1');
        if (result.success && result.data.nodes.length > 0) {
            graphData = result.data;
            drawGraph();
        }
    } catch (e) {
        console.error('Failed to load graph:', e);
    }
}

function resizeCanvas() {
    const canvas = graphCanvas;
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width || 360;
    canvas.height = (rect.height - 90) || 400;
    if (graphData.nodes.length > 0) drawGraph();
}

function drawGraph() {
    const canvas = graphCanvas;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    if (graphData.nodes.length === 0) {
        ctx.fillStyle = '#6b6b80';
        ctx.font = '13px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Chưa có dữ liệu Knowledge Graph', W / 2, H / 2);
        return;
    }

    // Color map
    const colorMap = {
        'VAN_BAN': '#a78bfa',
        'CHUONG': '#fbbf24',
        'DIEU_LUAT': '#818cf8',
        'KHAI_NIEM': '#34d399',
        'CHU_THE': '#fb923c',
        'HANH_VI': '#f87171',
        'QUYEN': '#60a5fa',
        'NGHIA_VU': '#a78bfa',
    };

    // Layout: circular
    const nodes = graphData.nodes;
    const edges = graphData.edges;
    const centerX = W / 2;
    const centerY = H / 2;
    const radius = Math.min(W, H) * 0.35;

    const nodePositions = {};
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
        nodePositions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle),
            node: node,
        };
    });

    // Draw edges
    ctx.lineWidth = 1;
    edges.forEach(edge => {
        const src = nodePositions[edge.source];
        const tgt = nodePositions[edge.target];
        if (!src || !tgt) return;

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
        ctx.stroke();
    });

    // Draw nodes
    nodes.forEach((node, i) => {
        const pos = nodePositions[node.id];
        if (!pos) return;

        const color = colorMap[node.type] || '#818cf8';
        const nodeRadius = node.type === 'DIEU_LUAT' ? 8 : 5;

        // Glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius + 4, 0, Math.PI * 2);
        ctx.fillStyle = color + '20';
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Label
        ctx.fillStyle = '#a0a0b8';
        ctx.font = '9px Inter';
        ctx.textAlign = 'center';
        const label = node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label;
        ctx.fillText(label, pos.x, pos.y + nodeRadius + 14);
    });
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
