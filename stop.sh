#!/bin/bash

ps -ef | grep "open_kf_app:app" | grep -v grep | awk '{print $2}' | xargs kill -9
