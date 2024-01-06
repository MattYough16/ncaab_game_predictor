#!/bin/sh
# Create a folder (named dmg) to prepare our DMG in (if it doesn't already exist).
mkdir -p dist/dmg
# Empty the dmg folder.
rm -r dist/dmg/*
# Copy the app bundle to the dmg folder.
cp -r "dist/NCAAB_Score_Predictor.app" dist/dmg
# If the DMG already exists, delete it.
test -f "dist/NCAAB_Score_Predictor.dmg" && rm "dist/NCAAB_Score_Predictor.dmg"
create-dmg \
  --volname "NCAAB_Score_Predictor" \
  --icon "NCAAB_Score_Predictor.app" 175 120 \
  --hide-extension "NCAAB_Score_Predictor.app" \
  --app-drop-link 425 120 \
  "dist/NCAAB_Score_Predictor.dmg" \
  "dist/dmg/"