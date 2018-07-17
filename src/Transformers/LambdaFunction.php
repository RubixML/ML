<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(callable $lambda)
    {
        $this->lambda = $lambda;
    }

    /**
     * Do nothing as this is a stateless transformer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        //
    }

    /**
     * Run the lamba function over the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        $samples = call_user_func($this->lambda, $samples);
    }
}
