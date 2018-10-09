<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;

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
class LambdaFunction implements Transformer, Online
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
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        //
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function update(DataFrame $dataframe) : void
    {
        //
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        $samples = call_user_func($this->lambda, $samples);
    }
}
