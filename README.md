# chirp-chirp
Bird Song Classifier model - Third year project for BSc Computer Science and Mathematics and the University of Manchester.

## Related Repositories
[**chirp-chirp iOS application**](https://github.com/Suvi1301/chirp-chirp-ios)

[**Project Report & Demo**](https://github.com/Suvi1301/chirp-chirp-report)

## Useful commands

### Run under waitress
```
waitress-serve --port=5000 web.app:APP
```

### Build docker image
```
docker build -t chirp-chirp .
```

### Run under docker

```
docker run -t -p 5000:5000 -e CHIRP_CHOSEN_MODEL=$CHIRP_CHOSEN_MODEL chirp-chirp
```
