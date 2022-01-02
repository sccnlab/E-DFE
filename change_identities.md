## Change Identities (You can change facial identities in MakeHuman manually)
### For changing skin, gender, race and facial shape in code, add those steps in makehuman_face.py (You need to add Geometries/Clothes, Geometries/Hair and Geometries/Teeth manually in MakeHuman) before setting AUs
1. Chose features from the material_path and facial_dict (see facial_animation/categories.txt)
2. Run test the features in material_path and facial_dict to pick the features you prefer
3. Use this line of code to set skin color  `G.app.selectedHuman.material = material.fromFile(# item in material_path)`
4. Use the code below to set facial shape
   ```bash
   Face_modifier = human.getModifier(# item in facial_dict)
   Face_modifier.setValue(custom value from -1 to 1)
   human.applyAllTargets()
    ```
   You need to change back to original face shape before changing to the next one
   ```bash
   Face_modifier = human.getModifier(# item in facial_dict)
   Face_modifier.setValue(0)
   human.applyAllTargets()
   ```

