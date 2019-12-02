# Model Persistence
Model persistence refers to the capability of an estimator to be trained and used to make predictions in processes other the current running process. Imagine that you trained a classifier to categorize comment posts and now you want to deploy it to a server to perform real-time inference on your website. Or, say you just finished training a model that took the whole day and you want to save it for later. Rubix ML allows you to handle both of these scenarios using [Persisters](./persiters/api.md) and [Persistable](persistable.md) objects.

### Persisters
Persisters are objects whose responsibility is to save and load model data to and from storage. For example, the [Filesystem](./persisters/filesystem.md) serializes and reconstitutes a persistable model from a location on a filesystem such as a local hard disk drive (HDD) or network attached storage (NAS).

**Example**

```php
use Rubix\ML\Persisters\Filesystem;

$persister = new Filesystem('example.model');

$estimator = $persister->load();

// Do something

$persister->save($estimator);
```

### Serialization
Very often a model will need to be serialized, or packaged into a discrete chunk of data, before it can be persisted. The same is true for loading a model which is serialization in reverse. Rubix ML is compatible with a number of portable serialization formats including the [Native](https://docs.rubixml.com/en/latest/persisters/serializers/native.html) PHP format as well as the [Igbinary](https://docs.rubixml.com/en/latest/persisters/serializers/binary.html) format. By knowing the format, you can transport models between systems.

> **Note:** Due to a limitation in PHP, anonymous classes and functions are not able to be deserialized. If you add anonymous classes or functions to the model, they must be given formal definitions before they can be persisted.

### The Persistent Model Meta-estimator
The [Persistent Model](persistent-model.md) meta-estimator is a model wrapper that uses the persistence subsystem to save and load model data. It provides the `save()` and `load()` methods for operating on the persistable learner that it wraps.

**Example**

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('example.model'));

// Do something

$estimator->save();
```