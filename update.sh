#!/bin/bash

CURRENT_TAG=$(git describe --exact-match HEAD 2>&1)
NEWEST_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))
if [[ $CURRENT_TAG == $NEWEST_TAG ]]
then
    echo "You are already using the latest release."
    exit 1
fi
git checkout $NEWEST_TAG