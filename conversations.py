"""
Conversation Manager - Handles persistent storage and management of past conversations
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

CONVERSATIONS_DIR = Path("./conversations_data")
CONVERSATIONS_FILE = CONVERSATIONS_DIR / "conversations.json"

def ensure_conversations_dir():
    """Create conversations directory if it doesn't exist"""
    CONVERSATIONS_DIR.mkdir(exist_ok=True)

def generate_title(first_message: str, max_length: int = 50) -> str:
    """Generate a conversation title from the first user message"""
    title = first_message.strip()
    if len(title) > max_length:
        title = title[:max_length].rsplit(' ', 1)[0] + "…"
    return title or "Untitled Conversation"

def load_conversations() -> Dict:
    """Load all conversations from JSON"""
    ensure_conversations_dir()
    if CONVERSATIONS_FILE.exists():
        try:
            with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return {}
    return {}

def save_conversations(data: Dict) -> None:
    """Save conversations to JSON"""
    ensure_conversations_dir()
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_conversation(first_user_message: str) -> str:
    """Create a new conversation and return its ID"""
    import uuid
    conv_id = str(uuid.uuid4())[:8]
    
    conversations = load_conversations()
    conversations[conv_id] = {
        "id": conv_id,
        "title": generate_title(first_user_message),
        "custom_title": None,  # User-provided custom title
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "pinned": False,
        "messages": [],
        "memory": []
    }
    save_conversations(conversations)
    return conv_id

def add_message_to_conversation(conv_id: str, role: str, content: str) -> None:
    """Add a message to a conversation"""
    conversations = load_conversations()
    if conv_id in conversations:
        conversations[conv_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        conversations[conv_id]["updated_at"] = datetime.now().isoformat()
        save_conversations(conversations)

def get_conversation(conv_id: str) -> Optional[Dict]:
    """Get a specific conversation"""
    conversations = load_conversations()
    return conversations.get(conv_id)

def get_all_conversations() -> List[Dict]:
    """Get all conversations, sorted by update time (newest first), with pinned at top"""
    conversations = load_conversations()
    conv_list = list(conversations.values())
    
    # Sort: pinned first, then by updated_at descending
    conv_list.sort(key=lambda x: (not x.get("pinned", False), x.get("updated_at", "")), reverse=True)
    return conv_list

def delete_conversation(conv_id: str) -> None:
    """Delete a conversation"""
    conversations = load_conversations()
    if conv_id in conversations:
        del conversations[conv_id]
        save_conversations(conversations)

def rename_conversation(conv_id: str, new_title: str) -> None:
    """Rename a conversation"""
    conversations = load_conversations()
    if conv_id in conversations:
        conversations[conv_id]["custom_title"] = new_title
        conversations[conv_id]["updated_at"] = datetime.now().isoformat()
        save_conversations(conversations)

def pin_conversation(conv_id: str, pinned: bool = True) -> None:
    """Pin or unpin a conversation"""
    conversations = load_conversations()
    if conv_id in conversations:
        conversations[conv_id]["pinned"] = pinned
        conversations[conv_id]["updated_at"] = datetime.now().isoformat()
        save_conversations(conversations)

def get_display_title(conversation: Dict) -> str:
    """Get the display title (custom if set, otherwise auto-generated)"""
    return conversation.get("custom_title") or conversation.get("title", "Untitled")

def search_conversations(query: str) -> List[Dict]:
    """Search conversations by title"""
    query_lower = query.lower()
    all_convs = get_all_conversations()
    return [
        c for c in all_convs 
        if query_lower in get_display_title(c).lower()
    ]

def filter_conversations(sort_by: str = "recent", pinned_only: bool = False) -> List[Dict]:
    """Filter and sort conversations"""
    convs = get_all_conversations()
    
    if pinned_only:
        convs = [c for c in convs if c.get("pinned", False)]
    
    if sort_by == "recent":
        convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    elif sort_by == "oldest":
        convs.sort(key=lambda x: x.get("updated_at", ""))
    elif sort_by == "alphabetical":
        convs.sort(key=lambda x: get_display_title(x).lower())
    
    return convs
