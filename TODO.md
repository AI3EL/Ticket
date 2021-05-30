
### Preprocessing

- Overall, when letters are not full (lack of ink), tesseract is very bad
- Recorrect homography using text lines ? Maybe local homography for each line ? 


### Custom Font

- Not urgent but could be nice: train on custom font: https://pretius.com/how-to-prepare-training-files-for-tesseract-ocr-and-improve-characters-recognition/
- maybe better source to training: https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html
- bold and not bold: might be useful to know who is category

### Better usage of tesseract

- List of unauthorized characters for instance
- pytesseract.run_and_get_output doesn't work (Aucun fichier ou dossier de ce type: '/tmp/tess_5m7e7n6r.')
- try tesserocr but does not seem to work

### Trailing dots

- Not detecting Fruits in 1b.jpg: doesn't seem to be the homography's fault, maybe trailing dots that cramp the line processing
- Detect category using trailing dots and bold font
- Shift kernel beginning a bit ?

### Other

- implement word distance: issue that we have abrevitaions
- Look at : https://github.com/phdenzel/pentaplex/blob/master/receipt.py
- nice googles: 'receipt ocr preprocessing'
- try line by line ocr ?

### Long term

- Know the category of each object to infer category in case it was skipped
- Curvy lines: https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv

### Ressources:
 
- Removing shadows: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
- https://tesseract-ocr.github.io/tessdoc/ImproveQuality
- https://groups.google.com/g/tesseract-ocr