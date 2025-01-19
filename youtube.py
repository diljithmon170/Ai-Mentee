const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const PORT = 3000;

// Replace with your YouTube Data API key
const API_KEY = 'YTB056556kvjnbhjfd016546323dnfkjshgjMVnavdbyoutgbrvhvuvxcjnajh';

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.set('view engine', 'ejs');
app.use(express.static('public'));

// Function to fetch video links
async function getVideoLinks(query, maxResults = 5) {
    const url = `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(
        query
    )}&type=video&maxResults=${maxResults}&key=${API_KEY}`;
    try {
        const response = await axios.get(url);
        const videos = response.data.items.map((item) => {
            const videoId = item.id.videoId;
            const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;
            return {
                title: item.snippet.title,
                url: videoUrl,
            };
        });
        return videos;
    } catch (error) {
        console.error('Error fetching videos:', error.message);
        return [];
    }
}

// Route: Home page
app.get('/', (req, res) => {
    res.render('index', { videos: [] });
});

// Route: Fetch videos
app.post('/search', async (req, res) => {
    const topic = req.body.topic;
    const videos = await getVideoLinks(topic);
    res.render('index', { videos });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
