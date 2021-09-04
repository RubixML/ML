<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\StopWordFilter;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class StopWordFilterBench
{
    protected const DATASET_SIZE = 10000;

    protected const STOP_WORDS = [
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it',
        'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they',
        'this', 'to', 'was', 'will', 'with',
    ];

    protected const SAMPLE_TEXT = "Machine learning is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as 'training data', in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.";

    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    public function setUp() : void
    {
        /** @var positive-int $k */
        $k = (int) (strlen(self::SAMPLE_TEXT) / 8);

        $samples = [];

        for ($i = 0; $i < self::DATASET_SIZE; ++$i) {
            $samples[] = str_split(self::SAMPLE_TEXT, $k);
        }

        $this->dataset = new Unlabeled($samples);
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply(new StopWordFilter(self::STOP_WORDS));
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function applyEmpty() : void
    {
        $this->dataset->apply(new StopWordFilter([]));
    }
}
