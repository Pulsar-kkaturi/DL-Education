# DL-Education
"딥러닝 강의 프로그래밍 실습 파일 모음"
by Jonghun Jeong
* **Contents**
    1. Information
    2. Project Description
    3. Code Description

---
## 1. Information

### 1.1. Author
* Main Developer: Jonghun Jeong
* Institution: Vuno Inc.
    - Medical Imaging Group
    - BLU3 (Brain & LUng 3D) Team
    - Position: Researcher
* Institution: SNU GSCST Dept. of Applied Bioengineering
    - Laboratory: BBL (Bioimaging and Biophotonics Laboratory)
    - Position: PhD Course Student
    - Research Interests: Application of deep learning to Optical Microscopy

---
## 2. Project Description
### 2.1. Python Lecture
"파이썬 실습 노트"
* Reference: 박응용, 위키독스, https://wikidocs.net/book/1

### 2.2. VisionDL Lecture
"비전 딥러닝 특강"
* 강의노트: https://pulsar-kkaturi.tistory.com/category/%EA%B5%90%EC%9C%A1%20%EB%A6%AC%EB%B7%B0/%EB%B9%84%EC%A0%84%20%EB%94%A5%EB%9F%AC%EB%8B%9D%20%ED%8A%B9%EA%B0%95
* Reference: 최건호, 한빛미디어, 2019년 06월 07일, 파이토치 첫걸음
* Reference: 프랑소와 숄레, 길벗, 2018년 10월 22일, 케라스 창시자에게 배우는 딥러닝

---
## 3. Code Description

### 3.1. Docker setting
* Dockerfile: 도커 환경 설정 파일
    - Docker-Base: nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
* .docker.sh: 도커 빌드 & 실행 Shell Script 파일

        bash .docker.sh {DATA_PATH} build # 도커 빌드
        bash .docker.sh {DATA_PATH} run # 도커 실행

* requirements.txt: python library list
