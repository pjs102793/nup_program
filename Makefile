APP_NAME = trtis_test
IMAGE_NAME = trtis_test

MODEL_VOLUME = /home/mkkwak3197/nup_program:/workspace/test

# Build and run the container
build:
	@echo 'build docker $(APP_NAME)'
	docker build --no-cache -t $(APP_NAME) . 

run:
	@echo 'run docker $(APP_NmAME)'
	docker run --rm -it --name="$(APP_NAME)" --net=host --ipc=host -v $(MODEL_VOLUME) --cpuset-cpus="48-63" --gpus all $(IMAGE_NAME)

stop:
	@echo 'stop docker $(APP_NAME)'
	docker stop $(APP_NAME)

rm:
	@echo 'rm docker $(APP_NAME)'
	docker rm -f $(APP_NAME)

rmi:
	@echo 'rmi docker $(APP_NAME)'
	docker rmi $(APP_NAME)
