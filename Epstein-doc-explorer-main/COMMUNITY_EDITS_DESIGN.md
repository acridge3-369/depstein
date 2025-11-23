# Community Edits System Design

## Overview

Allow users to propose corrections and provide context for relationship data without requiring authentication or modifying the source database.

## Architecture

### Two-Database Approach

1. **`document_analysis.db`** - Read-only source of truth
   - Contains all extracted relationships
   - Never modified by user input
   - Authoritative data

2. **`community_edits.db`** - User contributions
   - Edit proposals, votes, comments
   - Can be cleared/reset without data loss
   - Reputation-based moderation

### User Identification

**No accounts required** - Use browser fingerprinting:
- Combination of: User-Agent, screen resolution, timezone, canvas fingerprint
- Library: `@fingerprintjs/fingerprintjs`
- Purpose: Prevent duplicate votes, track reputation (optional future feature)
- Privacy: Fingerprints hashed, not stored raw

## Core Features

### 1. Edit Proposals

**Types of edits:**
- `identify_actor` - "Unknown Person A is actually John Smith"
- `identify_target` - "Unknown location is Palm Beach, FL"
- `correct_actor` - "This says 'Jeff' but should be 'Jeffrey Epstein'"
- `correct_target` - "Target misidentified, should be X"
- `add_context` - "Additional context about this relationship"
- `dispute` - "I don't think this relationship is accurate because..."

**Proposal workflow:**
1. User clicks "Suggest Edit" on a relationship in timeline
2. Modal opens with:
   - Current value (read-only)
   - Proposed value (editable)
   - Evidence text area (required, min 20 chars)
   - Optional: Link to supporting documents
   - Optional: Pseudonym (stored with edit)
3. Submit → Creates `edit_proposal` record with status `pending`
4. Appears as badge on relationship: "📝 1 edit proposed"

### 2. Voting System

**Upvote/Downvote:**
- Each fingerprint can vote once per proposal
- Vote changes allowed (switching from up to down or vice versa)
- Net score = upvotes - downvotes

**Auto-status updates (via trigger):**
- `net_score >= 5` → status = `accepted` (green badge)
- `net_score <= -3` → status = `rejected` (hidden by default)
- `total_votes >= 10` AND `|net_score| < 3` → status = `disputed` (yellow badge)

**Display in UI:**
- Proposals sorted by net_score DESC
- Show vote count: "↑ 12 ↓ 3"
- Current user's vote highlighted
- Accepted proposals shown with green checkmark

### 3. Discussion Threads

**Nested comments:**
- Each edit proposal has a comment thread
- Comments can reply to other comments (parent_comment_id)
- Comments also have upvote/downvote
- Sorted by net_score DESC (best comments first)

**Comment features:**
- Optional pseudonym
- Markdown support for formatting
- Timestamp (relative: "2 hours ago")
- "Reply" button for threading

### 4. Moderation

**Community flagging:**
- Users can flag edits/comments for:
  - Spam
  - Abuse/harassment
  - Misinformation
  - Duplicate
- One flag per user per item
- Auto-hide if flags > threshold (e.g., 5)

**Admin tools (future):**
- Dashboard showing flagged content
- Ability to delete/ban fingerprints
- Export/import community db for backup

## API Endpoints

### Edit Proposals

```typescript
// Get edits for a specific triple
GET /api/edits/triple/:tripleId
Response: {
  edits: EditProposal[],
  totalCount: number
}

// Create new edit proposal
POST /api/edits
Body: {
  tripleId: number,
  docId: string,
  editType: string,
  proposedValue: string,
  originalValue: string,
  evidenceText: string,
  supportingDocIds?: string[],
  submitterName?: string,
  submitterFingerprint: string
}

// Vote on edit
POST /api/edits/:editId/vote
Body: {
  vote: 1 | -1,
  voterFingerprint: string
}
```

### Comments

```typescript
// Get comments for edit
GET /api/edits/:editId/comments

// Post comment
POST /api/edits/:editId/comments
Body: {
  commentText: string,
  parentCommentId?: number,
  commenterName?: string,
  commenterFingerprint: string
}

// Vote on comment
POST /api/comments/:commentId/vote
Body: {
  vote: 1 | -1,
  voterFingerprint: string
}
```

### Moderation

```typescript
// Flag content
POST /api/moderation/flag
Body: {
  targetType: 'edit' | 'comment',
  targetId: number,
  flagReason: string,
  flaggerFingerprint: string
}
```

## UI Components

### 1. EditBadge Component
Shows on each relationship in timeline:
```jsx
<div className="edit-badge">
  {acceptedEdits > 0 && <span className="accepted">✓ {acceptedEdits}</span>}
  {pendingEdits > 0 && <span className="pending">📝 {pendingEdits}</span>}
  {disputedEdits > 0 && <span className="disputed">⚠️ {disputedEdits}</span>}
</div>
```

### 2. EditModal Component
Full-screen modal showing:
- Original relationship details
- All edit proposals (sorted by vote)
- Vote buttons for each proposal
- Discussion thread for each proposal
- "Propose New Edit" button

### 3. EditForm Component
Form for creating new edit:
- Edit type dropdown
- Proposed value input
- Evidence textarea (required, min 20 chars)
- Supporting docs multi-select (optional)
- Pseudonym input (optional)
- Character count for evidence

### 4. EditProposal Component
Single edit display:
```
┌─────────────────────────────────────────┐
│ [identify_actor] by JohnDoe123          │
│ 2 hours ago · ↑ 12 ↓ 3 (net: +9) ✓     │
├─────────────────────────────────────────┤
│ Unknown Person A → Jeffrey Epstein      │
│                                         │
│ Evidence:                               │
│ Cross-referencing with DOC-045 shows... │
│                                         │
│ Supporting docs: DOC-045, DOC-067       │
│                                         │
│ [↑ Upvote] [↓ Downvote] [💬 5 comments] │
└─────────────────────────────────────────┘
```

### 5. CommentThread Component
Nested comment display:
- Top-level comments first
- "Show replies (3)" for nested comments
- Vote buttons on each comment
- Reply button

## Data Blending Strategy

**Option A: Show edits as overlays (non-destructive)**
- Display original data
- Show accepted edits as suggestions: "Unknown Person A (suggested: Jeffrey Epstein)"
- User can toggle to see "community consensus" view

**Option B: Hybrid display**
- If `net_score >= 10` (high confidence), show proposed value by default
- Include badge: "Community edit (↑ 45)"
- Click to see original value and evidence

**Option C: Filter-based**
- Checkbox: "Show community-accepted edits"
- When enabled, replace values with accepted proposals
- Visual indicator that data is modified

**Recommendation: Option B** - Best balance of trust and transparency

## Privacy & Safety

### Anti-Spam Measures
1. Rate limiting per fingerprint:
   - Max 5 edit proposals per hour
   - Max 50 votes per hour
   - Max 20 comments per hour

2. Fingerprint validation:
   - Require fingerprint to be consistent
   - Flag suspicious patterns (VPN hopping, etc.)

3. Content validation:
   - Min/max length requirements
   - Profanity filter (configurable)
   - Link spam detection

### Data Retention
- Edit proposals: Keep indefinitely
- Rejected proposals: Hide after 30 days (but keep for reference)
- Flagged content: Hide immediately, review manually

### Export/Backup
- Daily backup of community_edits.db
- Export accepted edits as JSON for review
- Ability to seed new deployments with vetted edits

## Future Enhancements

### Reputation System
- Track fingerprint contributions
- Award badges: "Top Contributor", "Fact Checker", etc.
- Weighted voting (high-reputation users count more)

### AI-Assisted Validation
- Use Claude to evaluate edit proposals against evidence
- Auto-flag potentially incorrect edits
- Suggest related documents

### Batch Edit Mode
- Allow users to propose edits to multiple similar cases
- Example: "All instances of 'Unknown Person A' in DOC-XXX series are actually Y"

### Edit History
- Track when edits change status
- Show revision history for proposals
- Rollback mechanism

### Integration with Main DB
- Periodic review of highly-voted edits
- Option to "accept into canon" by re-running analysis with corrections
- Generate training data for improving LLM extraction

## Implementation Priority

**Phase 1 (MVP):**
1. Create community_edits.db schema
2. Add API endpoints for proposals and votes
3. Basic EditBadge and EditModal components
4. Browser fingerprinting integration

**Phase 2:**
1. Comment threads
2. Moderation/flagging
3. Auto-status updates
4. Advanced UI (nested threads, markdown)

**Phase 3:**
1. Reputation system
2. Batch edits
3. AI validation
4. Integration workflows

## Testing Strategy

### Unit Tests
- Vote counting logic
- Status update triggers
- Fingerprint validation
- Rate limiting

### Integration Tests
- Full edit workflow (propose → vote → comment)
- Concurrent voting (race conditions)
- Moderation pipeline

### Load Tests
- 1000 concurrent voters
- Large comment threads (100+ replies)
- Database size at 10k+ edits

### Security Tests
- Vote manipulation attempts
- SQL injection in text fields
- XSS in comment markdown
- Fingerprint spoofing

## Deployment Considerations

### Database Location
- Development: `community_edits.db` alongside main DB
- Production: Separate volume/backup schedule
- Render: Use Render's persistent disk for both DBs

### Backup Strategy
- Hourly snapshots of community_edits.db
- Weekly full backups
- Keep rejected/flagged content for audit

### Monitoring
- Track edit proposal volume
- Monitor voting patterns for manipulation
- Alert on high flag rates

### Rollout
- Beta test with limited users
- Gradual rollout (10% → 50% → 100%)
- A/B test UI variations
