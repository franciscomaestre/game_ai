
## A instalar

pip install -r requirements.txt

brew install ffmpge

## Resumen

Este código está preparado para funcionar correctamente con el Super Mario. Aún así, en esta etapa también ejecuta juegos de atari como el Space Invaders.

A futuro, se optimizará para ellos siguiendo el repo https://github.com/greydanus/baby-a3c

Para construir este repo, me basé en el repo https://github.com/uvipen/Super-mario-bros-A3C-pytorch/blob/master/README.md. A partir de su trabajo generé esta versión optimizada y refactorizada
 
## Errores conocidos

Si estás en un Mac, debes incluir en tu bashsrc export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES