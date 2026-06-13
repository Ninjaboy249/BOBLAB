# Live Streaming Feature Documentation

## Overview
The Football MatchLens application now includes a comprehensive live streaming feature that provides users with multiple ways to watch live football matches:
1. **Official YouTube Channel** - Direct access to live football matches
2. **Football Live Stream API** - Additional streaming links from various sources
3. **Custom URL Input** - Manual entry for any official stream

## Implementation Details

### 1. API Integration Function
**Function:** `fetch_live_stream_links(team_a, team_b)`

**Location:** `app.py` (lines 141-196)

**Purpose:** Fetches live streaming links for a specific match from the Football Live Stream API.

**Parameters:**
- `team_a` (str): Home team name
- `team_b` (str): Away team name

**Returns:** List of streaming links (can be dict objects with url/title/quality or plain URL strings)

**API Configuration:**
```python
API Endpoint: football-live-stream-api.p.rapidapi.com
API Key: ed35674ceemshdaf9b5b599f82cdp179c2djsn7c8722ead7dd
Headers:
  - x-rapidapi-key
  - x-rapidapi-host
  - Content-Type: application/json
```

**Features:**
- Automatically formats team names for API query (converts spaces to hyphens, lowercase)
- Handles multiple response formats (dict with links/streams/data, or direct list)
- Graceful error handling with user-friendly warnings
- Properly closes HTTP connections

### 2. Enhanced Watch Live Panel
**Function:** `render_watch_live_panel(team_a, team_b, score_a, score_b, elapsed, status, key_suffix)`

**Location:** `app.py` (lines 229-307)

**Enhancements:**

#### A. YouTube Live Football Channel (PRIMARY)
- **Channel URL:** https://www.youtube.com/channel/UCVDTKYExRDyv4k449OAkHqQ
- **Features:**
  - Direct "Watch Live" button to YouTube channel
  - Toggle to preview YouTube channel in-app
  - Embedded YouTube live stream player (520px height)
  - Official source for live football matches and highlights

#### B. Automatic Stream Discovery (SECONDARY)
- Fetches additional streaming links from Football Live Stream API
- Displays up to 5 streaming options
- Shows stream title, quality (if available), and watch button
- Each stream shown with title and quality indicator
- Direct "Watch" button for each stream
- Clean, organized layout using Streamlit columns

#### C. Manual URL Input (FALLBACK)
- Custom URL input field for any official stream
- LiveScore search as default fallback
- Preview in-app option for compatible streams
- Uses modern `st.iframe` (replaced deprecated `st.components.v1.iframe`)

## User Experience

### Watch Live Section Features:

1. **Live Match Card:**
   - Displays team names with flags
   - Shows current score
   - Displays match time and status

2. **📺 Live Football Channel (PRIMARY - NEW):**
   - **Official YouTube Channel** for live football matches
   - Direct "Watch Live" button to YouTube channel
   - Toggle to preview YouTube live stream in-app
   - Embedded YouTube player (520px height)
   - Channel: https://www.youtube.com/channel/UCVDTKYExRDyv4k449OAkHqQ

3. **🎥 Additional Live Streams (SECONDARY):**
   - Automatically fetched from Football Live Stream API
   - Multiple streaming options when available
   - One-click access to streams
   - Quality indicators (HD, SD, etc.)

4. **🔗 Custom Stream URL (FALLBACK):**
   - Manual URL input field
   - Open in new tab option
   - In-app preview toggle
   - Embedded iframe viewer (520px height)
   - Uses modern `st.iframe` API

## Usage Example

When a user navigates to the "Watch Live" section:

1. **Live Match Card** displays with current score and team flags
2. **YouTube Live Football Channel** appears first with:
   - Direct "Watch Live" button to open YouTube channel
   - Toggle to preview YouTube live stream in the app
3. **Additional Streams** automatically fetched from Football Live Stream API
4. **Custom URL Input** available as fallback option
5. All streams can be previewed in-app using embedded iframe

### User Flow:
```
User opens "Watch Live" section
    ↓
Sees live match card (teams, score, time)
    ↓
Primary Option: YouTube Live Football Channel
    → Click "Watch Live" → Opens YouTube in new tab
    → Toggle "Preview" → Embeds YouTube player in app
    ↓
Secondary Option: Additional API Streams (if available)
    → Multiple stream options with "Watch" buttons
    ↓
Fallback Option: Custom URL Input
    → Enter any official stream URL
    → Preview in app or open in new tab
```

## Error Handling

The implementation includes robust error handling:

- **API Connection Errors:** Displays warning, continues with manual input
- **Invalid Response:** Gracefully handles unexpected API responses
- **No Streams Found:** Silently falls back to manual URL input
- **Invalid URLs:** Warns user to enter valid http/https URLs

## Technical Notes

### API Response Handling
The function handles multiple response structures:
```python
# Dict with "links" key
{"links": [...]}

# Dict with "streams" key
{"streams": [...]}

# Dict with nested "data"
{"data": {"links": [...]}}

# Direct list
[...]
```

### Stream Link Formats
Supports both:
- **Dict format:** `{"url": "...", "title": "...", "quality": "..."}`
- **String format:** Direct URL strings

### Security Considerations
- Uses HTTPS connection
- API key stored in code (consider environment variables for production)
- URL validation before opening/embedding
- Warns about embedding restrictions

## Future Enhancements

Potential improvements:
1. Cache streaming links to reduce API calls
2. Add stream quality filtering
3. Implement stream availability checking
4. Add user preferences for preferred streams
5. Support for multiple match IDs/formats
6. Integration with more streaming APIs
7. Move API key to environment variables

## Testing

To test the feature:
1. Run the Streamlit app: `streamlit run app.py`
2. Navigate to a match prediction
3. Scroll to "Watch Live" section
4. **Test YouTube Channel Integration:**
   - Verify YouTube channel button appears
   - Click "Watch Live" button - should open YouTube channel
   - Toggle "Preview YouTube Channel" - should embed YouTube player
5. **Test API Streams:**
   - Verify additional streaming links appear (if available)
   - Test "Watch" buttons open streams
6. **Test Custom URL:**
   - Test manual URL input functionality
   - Test in-app preview toggle
7. **Verify Modern API:**
   - Confirm no deprecation warnings for `st.iframe`
   - Check console for any errors

## Dependencies

Required Python packages:
- `streamlit` - Web framework (uses modern `st.iframe` API)
- `http.client` - HTTP connections (built-in)
- `json` - JSON parsing (built-in)

**Note:** Updated to use `st.iframe` instead of deprecated `st.components.v1.iframe`

## API Documentation Reference

For more information about the Football Live Stream API:
- API Host: `football-live-stream-api.p.rapidapi.com`
- Documentation: Available on RapidAPI platform
- Rate Limits: Check RapidAPI dashboard for your plan

## Support

For issues or questions:
1. Check API key validity
2. Verify team name formatting
3. Review API response structure
4. Check network connectivity
5. Consult RapidAPI documentation

---

**Last Updated:** June 13, 2026
**Version:** 1.0
**Author:** Bob (AI Assistant)