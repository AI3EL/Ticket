### TODO:

- Ticket extraction for harder backgounds
- Think about automating evaluation process --> which metric, which format, etc ...
- improve preprocessing
- update parser
- treat title sections and others in line by line OCR

### Preprocessing

- Need vertical pad (for instance to detect commas in prices at the bottom), but it seems need less top pad than bottom pad
- Maybe forbids strange strings like -' or other but need to do a comprehensive study
- win_size = 51, k=0.05 good for the top of receipt2/3.jpg but leaves artifacts where there was shadow due to crumples at the bottom.
- Maybe look at sauvola multi-scale ?  https://hal.archives-ouvertes.fr/hal-02181880/document
- Use several y_shift in page ? Like cut page in slices of x inches 
- Test apply closing before or after binarization
- Test rm_shadow
- For binarization look at : https://github.com/brandonmpetty/Doxa
- asymetric vpad and remove what has already been read ? Maybe hard

### Custom Font

- Not urgent but could be nice: train on custom font: https://pretius.com/how-to-prepare-training-files-for-tesseract-ocr-and-improve-characters-recognition/
- maybe better source to training: https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html
- bold and not bold: might be useful to know who is category

### Better use of tesseract

- List of unauthorized characters or string (-', "#, ...)
- pytesseract.run_and_get_output doesn't work (Aucun fichier ou dossier de ce type: '/tmp/tess_5m7e7n6r.')
- try tesserocr but does not seem to work

### Other

- implement word distance: issue that we have abrevitaions
- Look at : https://github.com/phdenzel/pentaplex/blob/master/receipt.py
- nice googles: 'receipt ocr preprocessing'


### Ressources:
 
- Removing shadows: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
- https://tesseract-ocr.github.io/tessdoc/ImproveQuality
- https://groups.google.com/g/tesseract-ocr
- Curvy lines: https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv

### Long term

- Know the category of each object to infer category in case it was skipped