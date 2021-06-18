<?php

namespace Rubix\ML\Tests\Clusterers\Seeders;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Preset;
use PHPUnit\Framework\TestCase;

/**
 * @group Seeders
 * @covers \Rubix\ML\Clusterers\Seeders\Preset
 */
class PresetTest extends TestCase
{
    /**
     * @var \Rubix\ML\Clusterers\Seeders\Preset
     */
    protected $seeder;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->seeder = new Preset([
            ['foo', 14, 0.72],
            ['bar', 16, 0.92],
            ['beer', 21, 1.26],
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Preset::class, $this->seeder);
        $this->assertInstanceOf(Seeder::class, $this->seeder);
    }

    /**
     * @test
     */
    public function seed() : void
    {
        $expected = [
            ['foo', 14, 0.72],
            ['bar', 16, 0.92],
            ['beer', 21, 1.26],
        ];

        $seeds = $this->seeder->seed(Unlabeled::quick([['beef', 4, 13.0]]), 3);

        $this->assertCount(3, $seeds);

        $this->assertEquals($expected, $seeds);
    }
}
