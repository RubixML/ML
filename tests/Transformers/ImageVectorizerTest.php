<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @requires extension gd
 * @covers \Rubix\ML\Transformers\ImageVectorizer
 */
class ImageVectorizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\ImageVectorizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            [imagecreatefromjpeg('tests/test.jpg'), 'something else'],
        ]);

        $this->transformer = new ImageVectorizer(false);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ImageVectorizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->dataset->apply(new ImageResizer(3, 3));

        $this->dataset->apply($this->transformer);

        $outcome = [
            ['something else', 60, 35, 22, 102, 66, 53, 73, 44, 29, 79, 49, 36, 89, 57, 45, 56,
                32, 21, 85, 53, 44, 75, 49, 43, 34, 18, 12],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
