## Main APIs:

 - /api/query/general: For general legal questions without document context
 - /api/documents/upload: For uploading legal documents
 - /api/query/document: For querying against uploaded documents

## Additional Useful APIs:

 - /api/chat/history/{session_id}: Get chat history for a session
 - /api/session/{session_id}: Clear session and resources

## Key Features:

 - Session-based management using session_id
 - Temporary file handling for document uploads
 - Proper error handling and logging
 - CORS middleware for frontend integration
 - Pydantic models for request/response validation
 - Resource cleanup

## Improvements and Safety Features:

 - Temporary directory for each session
 - Resource cleanup on session deletion
 - Error handling and logging
 - Input validation
 - Session management

## For document analysis:

 - Upload document first using /api/documents/upload
 - Then query the document using /api/query/document

## For general legal queries:

Use /api/query/general directly


## Clean up resources using /api/session/{session_id} when done