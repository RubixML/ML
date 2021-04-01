# Model Persistence
Model persistence is the ability to save and subsequently load a learner's state in another process. Trained estimators can be used for real-time inference by loading the model onto a server or they can be saved to make predictions in batches offline at a later time. Estimators that implement the [Persistable](persistable.md) interface are able to have their internal state captured between processes. In addition, the library provides the [Persistent Model](persistent-model.md) meta-estimator that acts as a wrapper for persistable estimators.

## Serialization
Serialization occurs in between saving and loading a model and can be thought of as packaging the model's parameters. The data can be in a lightweight format such as with PHP's [Native](serializers/native.md) serializer or in a more robust format such as with the library's own [RBX](serializers/rbx.md) serializer. In the this example, we'll demonstrate how to encode a Persistable learner using the compressed RBX format and then how to unserialize the encoding.

```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Serializers\RBX;

$estimator = new RandomForest(100);

$serializer = new RBX();

$encoding = $serializer->serialize($estimator);

$estimator = $serializer->unserialize($encoding);
```

!!! note
    Due to a limitation in PHP, anonymous classes and functions (*closures*) are not able to be unserialized. Avoid adding anonymous classes or functions to an object that you intend to persist.

## Persisters
Persisters are objects that interface with your storage backend using the `save()` and `load()` API. In order to function properly, persisters must have read and write access to your storage system. In the example below, we'll use the [Filesystem](persisters/filesystem.md) persister to save the encoding we created in the first example.

```php
use Rubix\ML\Persisters\Filesystem;

$persister = new Filesystem('example.rbx');

$persister->save($encoding);
```

Then you can load the encoding from storage in another process.

```php
$encoding = $persister->load();
```

## Persistent Model Meta-estimator
The persistence subsystem can be interfaces at a low level with Serializer and Persister objects or it can be interacted with at a higher level using the [Persistent Model](persistent-model.md) meta-estimator. It is a decorator that provides `save()` and `load()` methods giving the estimator the ability to save and load itself.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('example.rbx'));

// Do something

$estimator->save();
```

## Persisting Transformers
In addition to Learners, the persistence subsystem can be used to save and load any Stateful transformer that implements the [Persistable](persistable.md) interface. In the example below we'll fit a transformer to a dataset and then save it to the [Filesystem](persisters/filesystem.md).

```php
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Persisters\Filesystem;

$transformer = new OneHotEncoder();

$serializer = new RBX();

$persister = new Filesystem('example.rbx');

$persister->save($serializer->serialize($transformer));
```

Then, to load the transformer in another process call the `unserialize()` method on the encoding returned by the persister's `load()` method.

```php
$transformer = $serializer->unserialize($persister->load());
```

## Caveats
Since model data are exported with the learner's current class definition in mind, problems may occur when loading a model using a different version of the library than the one it was trained and saved on. For example, when upgrading to a new version, there is a small chance that a previously saved learner may not be able to be deserialized if the model is not compatible with the learner's new class definition. For maximum interoperability, ensure that each system is running the same version of Rubix ML.
