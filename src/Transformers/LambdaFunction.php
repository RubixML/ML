<?php

namespace Rubix\ML\Transformers;

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
     * The user specified lambda function.
     *
     * @var callable
     */
    protected $lambda;

    /**
     * @param  callable  $lambda
     * @return void
     */
    public function __construct(callable $lambda)
    {
        $this->lambda = $lambda;
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        list($samples, $labels) = call_user_func($this->lambda, $samples, $labels);
    }
}
