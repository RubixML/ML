<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function call_user_func;

/**
 * Lambda Function
 *
 * Run a stateless lambda function (*anonymous* function) over the sample
 * matrix. The lambda function receives the sample matrix as an argument and
 * should return the transformed sample matrix.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LambdaFunction implements Transformer
{
    /**
     * A user-specified lambda function.
     *
     * @var callable
     */
    protected $lambda;

    /**
     * @param callable $lambda
     */
    public function __construct(callable $lambda)
    {
        $this->lambda = $lambda;
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
     * @param array[] $samples
     */
    public function transform(array &$samples) : void
    {
        $samples = call_user_func($this->lambda, $samples);
    }
}
