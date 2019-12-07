from .web import APP

APP.config['UPLOAD_FOLDER'] = 'web/uploads'
if __name__ == "__main__":
    APP.run()
