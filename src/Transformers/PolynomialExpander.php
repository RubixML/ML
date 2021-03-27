<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
     * @param list<list<mixed>> $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vector = [];

            foreach ($sample as $value) {
                $vector[] = $value;

                for ($i = 2; $i <= $this->degree; ++$i) {
                    $vector[] = $value ** $i;
                }
            }

            $sample = $vector;
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Polynomial Expander (degree: {$this->degree})";
    }
}
