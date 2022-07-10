<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageRandomRotationer;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @requires extension gd
 * @covers \Rubix\ML\Transformers\ImageRandomRotationer
 */
class ImageRandomRotationerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\ImageRandomRotationer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever'],
        ]);

        $this->transformer = new ImageRandomRotationer();
    }

    protected function tearDown() : void
    {
        if (file_exists('./test.png')) {
            unlink('./test.png');
        }
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ImageRandomRotationer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $mock = $this->createPartialMock(ImageRandomRotationer::class, ['getRotationDegrees']);
        $mock->method('getRotationDegrees')
            ->will($this->returnValue(180));

        $this->dataset->apply($mock);

        $sample = $this->dataset->sample(0);
        imagepng($sample[0], './test.png');

        $this->assertEquals(file_get_contents('./tests/test_rotated.png'), file_get_contents('./test.png'));
    }
}
