# chirp-chirp
Bird Song Classifier

## Useful commands

### Run under waitress
```
waitress-serve --port=5000 web.app:APP
```

### Build docker iamge
```
docker build -t chirp-chirp .
```

### Run under docker

```
docker run -t -p 5000:5000 -e CHIRP_CHOSEN_MODEL=$CHIRP_CHOSEN_MODEL chirp-chirp
```
