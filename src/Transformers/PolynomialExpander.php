<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;

/**
 * Polynomial Expander
 *
 * This Transformer will generate polynomial features up to and including the
 * specified degree. Polynomial expansion is often used to fit data that is
 * non-linear using a linear Estimator such as Ridge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PolynomialExpander implements Transformer, Online
{
    /**
     * The degree of the polynomials to generate. Higher order polynomials are
     * able to fit data better, however require extra features to be added
     * to the dataset.
     *
     * @var int
     */
    protected $degree;

    /**
     * @param  int  $degree
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $degree = 2)
    {
        if ($degree < 1) {
            throw new InvalidArgumentException('The degree of the polynomial'
                . ' must be greater than 0.');
        }

        $this->degree = $degree;
    }

    /**
     * Fit the transformer to the data.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->update($dataframe);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function update(DataFrame $dataframe) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataframe->types())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        $columns = count(reset($samples));

        foreach ($samples as &$sample) {
            $vector = [];

            for ($i = 0; $i < $columns; $i++) {
                for ($j = 1; $j <= $this->degree; $j++) {
                    $vector[] = $sample[$i] ** $j;
                }
            }

            $sample = $vector;
        }
    }
}
