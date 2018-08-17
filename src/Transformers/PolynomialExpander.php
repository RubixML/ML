<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
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
class PolynomialExpander implements Transformer
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }
    }

    /**
     * Transform each sample into a dense polynomial feature vector in the degree
     * given.
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
