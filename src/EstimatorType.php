<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

use function in_array;

/**
 * Estimator Type
 *
 * Estimator type enum.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class EstimatorType implements Stringable
{
    /**
     * The classifier estimator type code.
     *
     * @var int
     */
    public const CLASSIFIER = 1;

    /**
     * The regressor estimator type code.
     *
     * @var int
     */
    public const REGRESSOR = 2;

    /**
     * The clusterer estimator type code.
     *
     * @var int
     */
    public const CLUSTERER = 3;

    /**
     * The anomaly detector estimator type code.
     *
     * @var int
     */
    public const ANOMALY_DETECTOR = 4;

    /**
     * An array of human-readable string representations of the estimator types.
     *
     * @var literal-string[]
     */
    protected const TYPE_STRINGS = [
        self::CLASSIFIER => 'classifier',
        self::REGRESSOR => 'regressor',
        self::CLUSTERER => 'clusterer',
        self::ANOMALY_DETECTOR => 'anomaly detector',
    ];

    /**
     * An array of all the estimator type codes.
     *
     * @var list<int>
     */
    protected const ALL = [
        self::CLASSIFIER,
        self::REGRESSOR,
        self::CLUSTERER,
        self::ANOMALY_DETECTOR,
    ];

    /**
     * The integer-encoded estimator type.
     *
     * @var int
     */
    protected int $code;

    /**
     * Build a new estimator type object.
     *
     * @param int $code
     */
    public static function build(int $code) : self
    {
        return new self($code);
    }

    /**
     * Build a classifier type.
     *
     * @return self
     */
    public static function classifier() : self
    {
        return new self(self::CLASSIFIER);
    }

    /**
     * Build a regressor type.
     *
     * @return self
     */
    public static function regressor() : self
    {
        return new self(self::REGRESSOR);
    }

    /**
     * Build a clusterer type.
     *
     * @return self
     */
    public static function clusterer() : self
    {
        return new self(self::CLUSTERER);
    }

    /**
     * Build an anomaly detector type.
     *
     * @return self
     */
    public static function anomalyDetector() : self
    {
        return new self(self::ANOMALY_DETECTOR);
    }

    /**
     * @param int $code
     * @throws InvalidArgumentException
     */
    public function __construct(int $code)
    {
        if (!in_array($code, self::ALL)) {
            throw new InvalidArgumentException('Invalid type code.');
        }

        $this->code = $code;
    }

    /**
     * Return the integer-encoded estimator type.
     *
     * @return int
     */
    public function code() : int
    {
        return $this->code;
    }

    /**
     * Is the estimator type supervised?
     */
    public function isSupervised() : bool
    {
        return $this->isClassifier() or $this->isRegressor();
    }

    /**
     * Is it a classifier?
     *
     * @return bool
     */
    public function isClassifier() : bool
    {
        return $this->code === self::CLASSIFIER;
    }

    /**
     * Is it a regressor?
     *
     * @return bool
     */
    public function isRegressor() : bool
    {
        return $this->code === self::REGRESSOR;
    }

    /**
     * Is it a clusterer?
     *
     * @return bool
     */
    public function isClusterer() : bool
    {
        return $this->code === self::CLUSTERER;
    }

    /**
     * Is it an anomaly detector?
     *
     * @return bool
     */
    public function isAnomalyDetector() : bool
    {
        return $this->code === self::ANOMALY_DETECTOR;
    }

    /**
     * Return the estimator type as a string.
     *
     * @return string
     */
    public function __toString() : string
    {
        return self::TYPE_STRINGS[$this->code];
    }
}
