<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Polynomial Expander
 *
 * This transformer will generate polynomials up to and including the specified *degree* of each continuous feature.
 * Polynomial expansion is sometimes used to fit data that is non-linear using a linear estimator such as Ridge,
 * Logistic Regression, or Softmax Classifier.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PolynomialExpander implements Transformer
{
    /**
     * The degree of the polynomials to generate for each feature.
     *
     * @var int
     */
    protected int $degree;

    /**
     * @param int $degree
     * @throws InvalidArgumentException
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
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'expand']);
    }

    /**
     * Expand the continuous features of a sample.
     *
     * @param list<mixed> $sample
     */
    protected function expand(array &$sample) : void
    {
        $vector = [];

        foreach ($sample as $value) {
            $vector[] = $value;

            for ($exponent = 2; $exponent <= $this->degree; ++$exponent) {
                $vector[] = $value ** $exponent;
            }
        }

        $sample = $vector;
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
        return "Polynomial Expander (degree: {$this->degree})";
    }
}
