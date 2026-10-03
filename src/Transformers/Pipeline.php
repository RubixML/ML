<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persistable;
use Rubix\ML\Traits\AutotrackRevisions;

use function count;

/**
 * Pipeline
 *
 * Pipeline is a Transformer decorator capable of composing an arbitrarily
 * long series of Transformer middleware into a single unit. It fits the
 * stack to a training dataset, transforms incoming samples by streaming
 * them through each transformer in order, and — when updated — refines
 * the fitting of any Elastic transformers (or lazily fits any Stateful
 * ones that have not yet been seen) while still transforming the data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

    /**
     * A list of transformers to be applied in series.
     *
     * @var list<Transformer>
     */
    protected array $transformers = [
        //
    ];

    /**
     * @param list<Transformer> $transformers
     */
    public function __construct(array $transformers)
    {
        $this->transformers = $transformers;
    }

    /**
     * Return the data types that this transformer is compatible with. The
     * compatibility of the pipeline is defined by the compatibility of
     * the first transformer in the stack.
     *
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        $first = $this->transformers[0] ?? null;

        return $first ? $first->compatibility() : DataType::all();
    }

    /**
     * Fit the pipeline to a dataset. Every stateful transformer in the
     * stack is refit to the current dataset, then the samples are
     * transformed in place as they pass through the chain.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                $transformer->fit($dataset);
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Is the pipeline fitted? It is considered fitted when every
     * stateful transformer in the stack is fitted. An empty pipeline
     * is always considered fitted.
     *
     * @return bool
     */
    public function fitted() : bool
    {
        $fitted = true;

        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful and !$transformer->fitted()) {
                $fitted = false;
            }
        }

        return $fitted;
    }

    /**
     * Update the fitting of the pipeline with an incremental dataset.
     * Any elastic transformer in the stack will have its fitting refined.
     * Any stateful transformer that has not yet been fitted will be
     * lazily fitted. In all cases the samples are transformed in place
     * as they pass through the chain.
     *
     * @param Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Elastic) {
                $transformer->update($dataset);
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Transform the samples in place by streaming them through every
     * transformer in the stack in order.
     *
     * @param list<list<mixed>> $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($this->transformers as $transformer) {
            $transformer->transform($samples);
        }
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
            'transformers' => $this->transformers,
        ];
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
        return count($this->transformers) === 0
            ? 'Pipeline (transformers: [])'
            : 'Pipeline (' . Params::stringify($this->params()) . ')';
    }
}
