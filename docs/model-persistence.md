# Model Persistence
Model persistence is the ability to save and subsequently load a learner's state in another process. Trained estimators can be used for real-time inference by loading the model onto a server or they can be saved to make predictions offline at a later time. Estimators that implement the [Persistable](persistable.md) interface are able to have their internal state persisted between processes by a [Persister](persisters/api.md). In addition, the library provides the [Persistent Model](persistent-model.md) meta-estimator that acts as a wrapper for persistable estimators.

## Persisters
Persisters are objects that interface with your storage backend such as a filesystem or Redis database. They provide the `save()` and `load()` methods which take and receive persistable objects. In order to function properly, persisters must have read and write access to your storage system. In the example below, the [Filesystem](persisters/filesystem.md) persister loads a persistable estimator from the filesystem, such as the system's local hard drive, and then saves it after performing some task.

```php
use Rubix\ML\Persisters\Filesystem;

$persister = new Filesystem('example.model');

$estimator = $persister->load();

// Do something

$persister->save($estimator);
```

## Serialization
Serialization occurs in between saving and loading a model and can be thought of as packaging the model's parameters into a single contiguous blob of data. The data can be in byte-stream format such as with PHP's [Native](persisters/serializers/native.md) serializer or in binary format as with the [Igbinary](persisters/serializers/igbinary.md) serializer. In the next example, we demonstrate how to replace the default serializer of the [Filesystem](persisters/filesystem.md) persister with Igbinary format.

```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Igbinary;

$persister = new Filesystem('example.model', true, new Igbinary());
```

> **Note:** Due to a limitation in PHP, anonymous classes and functions (*closures*) are not able to be deserialized. Avoid adding anonymous classes or functions to an object that you wish to persist. Instead, use a named function or class.

## Persistent Model Meta-estimator
The [Persistent Model](persistent-model.md) meta-estimator is a wrapper that uses the persistence subsystem under the hood. It provides the `save()` and `load()` methods that give the estimator the ability to save and load itself.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('example.model'));

// Do something

$estimator->save();
```

## Caveats
Since model data are exported with the learner's current class definition in mind, problems may occur when loading a model using a different version of the library than the one it was trained and saved on. For example, when upgrading to a new version, there is a small chance that a previously saved learner may not be able to be deserialized if the model is not compatible with the learner's new class definition. For maximum interoperability, ensure that each system is running the same version of Rubix ML.