<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Persistable;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Serializers\Serializer;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Persistent Transformer
 *
 * The Persistent Transformer decorator gives a Stateful transformer two additional methods
 * (`save()` and `load()`) that allow it and the state of its fitting to be saved to and
 * retrieved from storage. It uses Persister objects to interface with various storage
 * backends such as the Filesystem. Pipelines are the most common use case, but any
 * Persistable Stateful transformer may be decorated.
 *
 * The decorator is a runtime object. Unlike the transformer it decorates, it is not itself
 * Persistable and therefore cannot be serialized, and the storage coordinates it carries
 * never leak into the transformer's saved state. Since the decorator delegates to the very
 * same transformer instance it was constructed with, any fitting performed through it is
 * captured by the next call to `save()`.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PersistentTransformer implements Transformer, Stateful, Elastic
{
    /**
     * The persistable base transformer.
     *
     * @var Stateful
     */
    protected Stateful $base;

    /**
     * The persister used to interface with the storage layer.
     *
     * @var Persister
     */
    protected Persister $persister;

    /**
     * The object serializer.
     *
     * @var Serializer
     */
    protected Serializer $serializer;

    /**
     * Factory method to restore the transformer from persistence.
     *
     * @param Persister $persister
     * @param Serializer|null $serializer
     * @throws InvalidArgumentException
     * @return self
     */
    public static function load(Persister $persister, ?Serializer $serializer = null) : self
    {
        $serializer ??= new RBX();

        $base = $serializer->deserialize($persister->load());

        if (!$base instanceof Stateful) {
            throw new InvalidArgumentException('Persisted object must'
                . ' implement the Stateful interface.');
        }

        return new self($base, $persister, $serializer);
    }

    /**
     * @param Stateful $base
     * @param Persister $persister
     * @param Serializer|null $serializer
     */
    public function __construct(Stateful $base, Persister $persister, ?Serializer $serializer = null)
    {
        $this->base = $base;
        $this->persister = $persister;
        $this->serializer = $serializer ?? new RBX();
    }

    /**
     * Return the data types that the transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'base' => $this->base,
            'persister' => $this->persister,
            'serializer' => $this->serializer,
        ];
    }

    /**
     * Return the base transformer instance.
     *
     * @return Stateful
     */
    public function base() : Stateful
    {
        return $this->base;
    }

    /**
     * Fit the transformer to a dataset. The base transformer is fitted in place — it is the
     * very same instance that `save()` will later serialize — so the fitting performed
     * through the decorator is captured by the next save.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $this->base->fit($dataset);
    }

    /**
     * Has the transformer been fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return $this->base->fitted();
    }

    /**
     * Update the fitting of the transformer with an incremental dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     */
    public function update(Dataset $dataset) : void
    {
        if (!$this->base instanceof Elastic) {
            throw new RuntimeException('Base Transformer must'
                . ' implement the Elastic interface.');
        }

        $this->base->update($dataset);
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     */
    public function transform(array &$samples) : void
    {
        $this->base->transform($samples);
    }

    /**
     * Save the transformer and the state of its fitting to storage.
     *
     * @throws RuntimeException
     */
    public function save() : void
    {
        if (!$this->base instanceof Persistable) {
            throw new RuntimeException('Base Transformer is not persistable.');
        }

        $encoding = $this->serializer->serialize($this->base);

        $this->persister->save($encoding);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Persistent Transformer (' . Params::stringify($this->params()) . ')';
    }
}
