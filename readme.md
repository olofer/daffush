# daffush

Multi-purpose (note-to-self) demonstration of the DDPM (Denoising Diffusion Probabilistic Model) algorithm, building a browser application with `dash`, using session caching, and fitting least-squares models with Random Fourier Features (RFFs).

## Usage

The `dash` development server can be started like this:
```
python3 daffush.py
```
Then point a browser to the (local) address/port being served.

Another option is to start this way:
```
gunicorn daffush:server --workers 1
```
