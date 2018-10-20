import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'nocheckcertificate': True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=y6Sxv-sUYtM'])
    ydl.download(['https://www.youtube.com/watch?v=ZZ5LpwO-An4'])
    ydl.download(['https://www.youtube.com/watch?v=vCadcBR95oU'])
    ydl.download(['https://www.youtube.com/watch?v=MYxAiK6VnXw'])
    ydl.download(['https://www.youtube.com/watch?v=T7e2egh0H1Y'])
