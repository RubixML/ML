# System Architecture
The high level API is designed around a few key abstractions and their corresponding types and interfaces. In addition, Rubix ML employs various mid and low level subsystems that power many of the learners. This layered architecture allows for power and flexibility while keeping the public interface simple and straightforward.

## General Architecture
From the perspective of data flowing in and out of a machine learning system, there are a number of components that the user *may* interact with. These include [Dataset](./datasets/api.md) objects, [Transformers](./transformers/api.md), [Estimators](estimator.md), and Meta-estimators. Starting from the top, the illustration below shows the path of data from input features to prediction within the library.

![Rubix ML System Architecture](https://raw.githubusercontent.com/RubixML/RubixML/master/docs/img/rubix-ml-system-architecture.svg?sanitize=true)

## Subsystems
Under the hood, Rubix ML utilizes a number of modular subsystems that are highly optimized for their purpose such as the graph, neural network, SVM, and tensor computing subsystems. Some mid and low level subsystems also run as optional PHP extensions.