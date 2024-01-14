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
     * @var ImageVectorizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
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
        $dataset = Unlabeled::quick([
            [imagecreatefrompng('tests/test.png'), 'something else'],
        ]);

        $dataset->apply(new ImageResizer(3, 3));

        $dataset->apply($this->transformer);

        $expected = [
            ['something else', 46, 51, 66, 130, 135, 134, 118, 119, 116, 25, 26, 45, 149, 154, 154, 180,
                183, 170, 39, 39, 54, 77, 80, 89, 141, 140, 132],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1.0);
    }
}
