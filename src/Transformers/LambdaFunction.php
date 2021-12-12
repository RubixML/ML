<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * Lambda Function
 *
 * Run a stateless lambda function over the samples in a dataset. The function receives three arguments - the sample to be
 * transformed, its row offset in the dataset, and a user-defined outside context variable that can be used to hold state.
 *
 * **Note:** If the transformation results in a change in dimensionality, the change must be consistent for each sample.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LambdaFunction implements Transformer
{
    /**
     * The function to call over the samples in the dataset.
     *
     * @var callable(mixed[],string|int,mixed):void
     */
    protected $callback;

    /**
     * The outside context that gets injected into the callback function on each call.
     *
     * @var mixed
     */
    protected $context;

    /**
     * @param callable(mixed[],string|int,mixed):void $callback
     * @param mixed $context
     */
    public function __construct(callable $callback, $context = null)
    {
        $this->callback = $callback;
        $this->context = $context;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, $this->callback, $this->context);
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
        return 'Lambda Function';
    }
}
