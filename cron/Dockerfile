FROM alpine:latest

RUN apk add --no-cache curl bash busybox-suid

COPY send_request.sh /usr/local/bin/send_request.sh

RUN chmod +x /usr/local/bin/send_request.sh

COPY cronjob /etc/crontabs/root

CMD ["crond", "-f", "-l", "2"]
