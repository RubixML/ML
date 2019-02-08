<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;

/**
 * Polynomial Expander
 *
 * This transformer will generate polynomials up to and including the
 * specified degree of each feature column. Polynomial expansion is sometimes
 * used to fit data that is non-linear using a linear estimator such as Ridge
 * or Logistic Regression.
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
     * @param int $degree
     * @throws \InvalidArgumentException
     */
    public function __construct(int $degree = 2)
    {
        if ($degree < 1) {
            throw new InvalidArgumentException('The degree of the polynomial'
                . " must be greater than 0, $degree given.");
        }

        $this->degree = $degree;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vector = [];

            foreach ($sample as $feature) {
                for ($i = 1; $i <= $this->degree; $i++) {
                    $vector[] = $feature ** $i;
                }
            }

            $sample = $vector;
        }
    }
}
