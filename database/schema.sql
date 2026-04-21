-- ==============================================
-- IT Law Chatbot - Database Schema
-- XAMPP MySQL
-- ==============================================

CREATE DATABASE IF NOT EXISTS it_law_chatbot
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE it_law_chatbot;

-- ----------------------------
-- 1. Law Documents (metadata per legal document)
-- ----------------------------
CREATE TABLE IF NOT EXISTS law_documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ten_van_ban VARCHAR(500) NOT NULL,
    so_hieu VARCHAR(100),
    so_vbhn VARCHAR(100),
    loai_van_ban VARCHAR(100),
    co_quan_ban_hanh VARCHAR(200),
    ngay_ban_hanh VARCHAR(50),
    ngay_hieu_luc VARCHAR(50),
    ngay_het_hieu_luc VARCHAR(50),
    trang_thai VARCHAR(50),
    sua_doi_boi TEXT,
    ban_su_dung TEXT,
    nhom VARCHAR(200),
    ghi_chu TEXT,
    source_file VARCHAR(300),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_so_hieu (so_hieu)
) ENGINE=InnoDB;

-- ----------------------------
-- 2. Document Chunks (from smart_chunker)
-- ----------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chunk_id VARCHAR(50) NOT NULL,
    chunk_sub_index INT DEFAULT 0,
    chunk_total_sub INT DEFAULT 1,
    chunk_tier INT DEFAULT 1,
    content TEXT NOT NULL,
    context_text TEXT,
    chuong_so VARCHAR(50),
    chuong_ten VARCHAR(300),
    muc_so VARCHAR(50),
    muc_ten VARCHAR(300),
    dieu_so VARCHAR(50),
    dieu_ten VARCHAR(500),
    is_repealed BOOLEAN DEFAULT FALSE,
    is_truncated BOOLEAN DEFAULT FALSE,
    do_dai_chunk INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES law_documents(id) ON DELETE CASCADE,
    INDEX idx_chunk_id (chunk_id),
    INDEX idx_dieu (document_id, dieu_so),
    INDEX idx_chuong (document_id, chuong_so)
) ENGINE=InnoDB;

-- ----------------------------
-- 3. Chunk Embeddings (vectors as BLOB)
-- ----------------------------
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    chunk_id INT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    model_name VARCHAR(200) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ----------------------------
-- 4. Knowledge Graph - Entities (Nodes)
-- ----------------------------
CREATE TABLE IF NOT EXISTS kg_entities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entity_id VARCHAR(200) NOT NULL UNIQUE,
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    description TEXT,
    properties JSON,
    chunk_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE SET NULL,
    INDEX idx_entity_type (entity_type),
    INDEX idx_entity_id (entity_id)
) ENGINE=InnoDB;

-- ----------------------------
-- 5. Knowledge Graph - Relationships (Edges)
-- ----------------------------
CREATE TABLE IF NOT EXISTS kg_relationships (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_entity_id VARCHAR(200) NOT NULL,
    target_entity_id VARCHAR(200) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
    INDEX idx_source (source_entity_id),
    INDEX idx_target (target_entity_id),
    INDEX idx_rel_type (relationship_type),
    UNIQUE KEY uk_relationship (source_entity_id, target_entity_id, relationship_type)
) ENGINE=InnoDB;

-- ----------------------------
-- 6. Conversations
-- ----------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(500) DEFAULT 'Cuộc hội thoại mới',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ----------------------------
-- 7. Messages
-- ----------------------------
CREATE TABLE IF NOT EXISTS messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sources JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    INDEX idx_conv_msg (conversation_id, created_at)
) ENGINE=InnoDB;
