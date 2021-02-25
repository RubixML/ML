<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function in_array;

/**
 * Conduit
 *
 * Conduits allow you to abstract a series of transformations into a single higher-order transformation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Conduit implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

    /**
     * The series of transformers to apply.
     *
     * @var list<\Rubix\ML\Transformers\Transformer>
     */
    protected $transformers = [
        //
    ];

    /**
     * The data types that the conduit is compatible with.
     *
     * @var list<\Rubix\ML\DataType>
     */
    protected $compatibility;

    /**
     * @param list<\Rubix\ML\Transformers\Transformer> $transformers
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $transformers)
    {
        $compatibility = [];

        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Transformer must implement the transformer interface.');
            }

            foreach ($transformer->compatibility() as $type) {
                if (!in_array($type, $compatibility)) {
                    $compatibility[] = $type;
                }
            }
        }

        $this->transformers = $transformers;
        $this->compatibility = $compatibility;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                if (!$transformer->fitted()) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                $transformer->fit($dataset);

                $dataset->apply($transformer);
            }
        }
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Elastic) {
                $transformer->update($dataset);

                $dataset->apply($transformer);
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<array> $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($this->transformers as $transformer) {
            $transformer->transform($samples);
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Conduit (transformers: ' . Params::toString($this->transformers) . ')';
    }
}
