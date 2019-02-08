<?php

namespace Rubix\ML\CrossValidation\Reports;

use Rubix\ML\Estimator;
use InvalidArgumentException;

/**
 * Contingency Table
 *
 * A Contingency Table is used to display the frequency distribution of class
 * labels among a clustering of samples.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ContingencyTable implements Report
{
    /**
     * The classes to compare in the table.
     *
     * @var array|null
     */
    protected $classes;

    /**
     * @param  array|null  $classes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?array $classes = null)
    {
        if (is_array($classes)) {
            $classes = array_unique($classes);

            foreach ($classes as $class) {
                if (!is_string($class) and !is_int($class)) {
                    throw new InvalidArgumentException('Class type must be a'
                        . ' string or integer, ' . gettype($class) . ' found.');
                }
            }
        }

        $this->classes = $classes;
    }

    /**
     * The estimator types that this report is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            Estimator::CLUSTERER,
        ];
    }

    /**
     * Generate the report.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @throws \InvalidArgumentException
     * @return array
     */
    public function generate(array $predictions, array $labels) : array
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('The number of labels'
                . ' must equal the number of predictions.');
        }

        if (is_null($this->classes)) {
            $classes = array_unique($labels);
        } else {
            $classes = $this->classes;
        }

        $clusters = array_unique($predictions);

        $table = array_fill_keys($clusters, array_fill_keys($classes, 0));

        $included = array_flip($classes);

        foreach ($labels as $i => $label) {
            if (isset($included[$label])) {
                $table[$predictions[$i]][$label]++;
            }
        }

        return $table;
    }
}
