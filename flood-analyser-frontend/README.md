# 🌊 Flood Detection System

A lean and efficient MVP for flood risk assessment that allows users to enter coordinates or upload images for AI-powered analysis.

## Features

- **Coordinate Analysis**: Enter latitude and longitude for location-based flood risk assessment
- **Image Analysis**: Upload terrain photos for AI-powered visual analysis
- **Interactive Map**: Visualize locations with risk overlay
- **AI-Powered**: Uses Google's Gemini AI for intelligent analysis
- **Clean UI**: Modern, responsive design with shadcn/ui components

## Quick Start

### Option 1: Start Both Servers Together (Recommended)

```bash
# Set up Gemini API key
cd backend
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# Start both servers
./start-dev.sh
```

Access the app at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Option 2: Start Servers Separately

#### Backend Setup
```bash
cd backend
python3 -m pip install -r requirements.txt
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
python3 start.py
```

#### Frontend Setup
```bash
npm install
npm run dev
```

## How to Use

1. **Coordinate Analysis**: Enter latitude and longitude coordinates
2. **Image Analysis**: Upload terrain photos for AI assessment
3. **View Results**: See risk level, elevation, distance from water, and recommendations
4. **Interactive Map**: Visualize the location with risk overlay

## API Endpoints

- `POST /api/analyze/coordinates` - Analyze flood risk by coordinates
- `POST /api/analyze/image` - Analyze flood risk by image upload
- `GET /health` - Health check

## Technical Stack

### Frontend
- Next.js 15 with App Router
- TypeScript
- Tailwind CSS + shadcn/ui
- Google Maps JavaScript API

### Backend
- FastAPI
- Google Gemini AI
- Python 3.9+
- Pydantic for validation

## Project Structure

```
flood-analyser/
├── app/page.tsx           # Main flood detection component
├── backend/main.py        # FastAPI application with Gemini AI
├── backend/start.py       # Startup script
├── start-dev.sh          # Development script
└── README.md             # Documentation
```

## Setup Requirements

1. **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Google Maps API Key** (optional): For interactive map functionality

## Development

```bash
# Quick start
./start-dev.sh

# Individual development
npm run dev          # Frontend
cd backend && python3 start.py  # Backend
```

The app is now lean and focused on the core flood detection functionality with AI-powered analysis.
