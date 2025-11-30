# Railway Deployment Guide

## Prerequisites
- Railway account (you already have one ✅)
- Model file: `sokoban_diffusion.pth` (3.6MB)

## Deployment Steps

### Option 1: Deploy via Railway Dashboard (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "feat: prepare for Railway deployment"
   git push origin main
   ```

2. **Create new Railway project**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `sokoflow` repository

3. **Upload model file**
   - In Railway dashboard, go to your project
   - Click on "Variables" tab
   - Or use "Files" tab to upload `sokoban_diffusion.pth`
   - **Alternative**: Use Railway CLI (see Option 2)

4. **Set environment variables** (if needed)
   - Railway auto-detects Python apps
   - PORT is automatically set by Railway

5. **Deploy**
   - Railway will auto-detect `requirements.txt` and `Procfile`
   - Build will start automatically
   - Check logs if there are issues

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI** (if not already installed)
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Initialize Railway project**
   ```bash
   railway init
   railway link  # Link to existing project or create new one
   ```

3. **Upload model file**
   ```bash
   railway up sokoban_diffusion.pth
   ```
   Or use the dashboard to upload files.

4. **Deploy**
   ```bash
   railway up
   ```

### Option 3: Include model in git (Quick but not ideal)

If you want to quickly test deployment:

1. **Temporarily allow model in git**
   ```bash
   git add -f sokoban_diffusion.pth
   git commit -m "chore: include model for deployment"
   git push
   ```

2. **Deploy on Railway**
   - Railway will pull from GitHub
   - Model will be included in deployment

3. **Remove from git later** (optional)
   ```bash
   git rm --cached sokoban_diffusion.pth
   git commit -m "chore: remove model from git"
   ```

## Important Notes

- **Model file is required**: The app needs `sokoban_diffusion.pth` to run
- **File size**: 3.6MB is within Railway's limits
- **Build time**: First build may take 5-10 minutes (installing PyTorch)
- **Memory**: Railway free tier should be sufficient, but monitor usage

## Troubleshooting

### Model not found error
- Ensure `sokoban_diffusion.pth` is in the project root
- Check Railway file system or upload via dashboard

### Build fails
- Check Railway logs for specific errors
- Ensure `requirements.txt` has correct versions
- PyTorch installation can be slow - be patient

### App crashes
- Check Railway logs
- Verify PORT environment variable is set (Railway does this automatically)
- Ensure model file is accessible

## After Deployment

1. Railway will provide a URL like: `https://your-app.railway.app`
2. Open it in browser to test
3. Monitor logs in Railway dashboard

## Training Configuration

By default, Railway uses **200 epochs** for faster deployment. You can customize this:

**Option 1: Via Railway Dashboard (Environment Variables)**
- Go to your Railway project → Variables tab
- Add `TRAIN_EPOCHS=500` for full training (better quality, slower)
- Add `TRAIN_USE_VALIDATION=true` to enable validation split

**Option 2: Via Railway CLI**
```bash
railway variables set TRAIN_EPOCHS=500
railway variables set TRAIN_USE_VALIDATION=true
```

**Trade-offs:**
- **200 epochs**: ~10-20 min, working model, faster deployment
- **500 epochs**: ~30-60 min, better quality, slower first deployment
- **Validation**: Adds ~20% time but helps detect overfitting

## Cost

- Railway free tier: $5/month credit
- This app should fit within free tier limits
- Monitor usage in Railway dashboard
- Training time counts toward compute usage

