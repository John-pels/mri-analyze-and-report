"use client"

import type React from "react"

import { useRef, useEffect, useState } from "react"
import {
    ZoomIn,
    ZoomOut,
    RotateCw,
    RotateCcw,
    Minimize,
    RefreshCw,
    Expand,
    ScanSearch,
    MoreHorizontal,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"
import { useMobile } from "@/hooks/use-mobile"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface Region {
    id: number
    location: {
        x: number
        y: number
        width: number
        height: number
    }
    confidence: string
    classification: string
}

interface ImageViewerProps {
    src: string
    alt: string
    regions: Region[]
}

export function ImageViewer({ src, alt, regions = [] }: ImageViewerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [zoom, setZoom] = useState(1)
    const [rotation, setRotation] = useState(0)
    const [isDragging, setIsDragging] = useState(false)
    const [position, setPosition] = useState({ x: 0, y: 0 })
    const [startPosition, setStartPosition] = useState({ x: 0, y: 0 })
    const [fullscreen, setFullscreen] = useState(false)
    const isMobile = useMobile()

    // Load and render the image
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const image = new Image()
        image.crossOrigin = "anonymous" // Prevent CORS issues
        image.src = src

        image.onload = () => {
            // Set canvas dimensions to match image
            canvas.width = image.width
            canvas.height = image.height

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height)

            // Apply transformations
            ctx.save()
            ctx.translate(canvas.width / 2, canvas.height / 2)
            ctx.rotate((rotation * Math.PI) / 180)
            ctx.scale(zoom, zoom)
            ctx.translate(-image.width / 2, -image.height / 2)

            // Draw image
            ctx.drawImage(image, 0, 0, image.width, image.height)

            // Reset transformations for drawing regions
            ctx.restore()

            // Draw regions of interest
            if (regions.length > 0) {
                // Apply transformations for regions
                ctx.save()
                ctx.translate(canvas.width / 2, canvas.height / 2)
                ctx.rotate((rotation * Math.PI) / 180)
                ctx.scale(zoom, zoom)
                ctx.translate(-image.width / 2, -image.height / 2)

                regions.forEach((region) => {
                    const { x, y, width, height } = region.location

                    // Draw rectangle
                    ctx.strokeStyle = getColorByConfidence(Number.parseFloat(region.confidence))
                    ctx.lineWidth = 2 / zoom // Adjust line width based on zoom
                    ctx.strokeRect(x, y, width, height)

                    // Draw label
                    ctx.fillStyle = getColorByConfidence(Number.parseFloat(region.confidence))
                    ctx.fillRect(x, y - 20 / zoom, width, 20 / zoom)

                    ctx.fillStyle = "#ffffff"
                    ctx.font = `${12 / zoom}px sans-serif`
                    ctx.fillText(`${region.classification} (${region.confidence})`, x + 5 / zoom, y - 5 / zoom)
                })

                ctx.restore()
            }
        }
    }, [src, zoom, rotation, regions])

    // Handle mouse events for panning
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        setIsDragging(true)
        setStartPosition({
            x: e.clientX - position.x,
            y: e.clientY - position.y,
        })
    }

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDragging) return

        setPosition({
            x: e.clientX - startPosition.x,
            y: e.clientY - startPosition.y,
        })
    }

    const handleMouseUp = () => {
        setIsDragging(false)
    }

    const handleMouseLeave = () => {
        setIsDragging(false)
    }

    // Handle zoom controls
    const zoomIn = () => {
        setZoom((prev) => Math.min(prev + 0.1, 3))
    }

    const zoomOut = () => {
        setZoom((prev) => Math.max(prev - 0.1, 0.5))
    }

    // Handle rotation controls
    const rotateClockwise = () => {
        setRotation((prev) => (prev + 90) % 360)
    }

    const rotateCounterClockwise = () => {
        setRotation((prev) => (prev - 90 + 360) % 360)
    }

    // Reset view
    const resetView = () => {
        setZoom(1)
        setRotation(0)
        setPosition({ x: 0, y: 0 })
    }

    // Toggle fullscreen
    const toggleFullscreen = () => {
        setFullscreen(!fullscreen)
    }

    // Helper function to get color based on confidence
    const getColorByConfidence = (confidence: number): string => {
        if (confidence >= 0.9) return "#10b981" // Green for high confidence
        if (confidence >= 0.7) return "#f59e0b" // Yellow for medium confidence
        return "#ef4444" // Red for low confidence
    }

    // Render the compact controls for small screens
    const renderCompactControls = () => (
        <div className="flex items-center justify-center space-x-2">
            <Button variant="outline" size="icon" onClick={zoomOut}>
                <ZoomOut className="h-4 w-4" />
            </Button>

            <Slider
                value={[zoom * 100]}
                min={50}
                max={300}
                step={10}
                className="w-16 sm:w-20"
                onValueChange={(value) => setZoom(value[0] / 100)}
            />

            <Button variant="outline" size="icon" onClick={zoomIn}>
                <ZoomIn className="h-4 w-4" />
            </Button>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="icon">
                        <MoreHorizontal className="h-4 w-4" />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={rotateCounterClockwise}>
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Rotate Left
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={rotateClockwise}>
                        <RotateCw className="h-4 w-4 mr-2" />
                        Rotate Right
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={resetView}>
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Reset View
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={toggleFullscreen}>
                        {fullscreen ? (
                            <>
                                <Minimize className="h-4 w-4 mr-2" />
                                Exit Fullscreen
                            </>
                        ) : (
                            <>
                                <Expand className="h-4 w-4 mr-2" />
                                Fullscreen
                            </>
                        )}
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    )

    // Render the full controls for larger screens
    const renderFullControls = () => (
        <div className="flex items-center space-x-2">
            <span className="text-xs text-muted-foreground mr-2 hidden md:inline-flex items-center">
                <ScanSearch className="h-3 w-3 mr-1" /> Image Controls:
            </span>
            <Button variant="outline" size="icon" onClick={zoomOut}>
                <ZoomOut className="h-4 w-4" />
            </Button>

            <Slider
                value={[zoom * 100]}
                min={50}
                max={300}
                step={10}
                className="w-24"
                onValueChange={(value) => setZoom(value[0] / 100)}
            />

            <Button variant="outline" size="icon" onClick={zoomIn}>
                <ZoomIn className="h-4 w-4" />
            </Button>

            <Button variant="outline" size="icon" onClick={rotateCounterClockwise}>
                <RotateCcw className="h-4 w-4" />
            </Button>

            <Button variant="outline" size="icon" onClick={rotateClockwise}>
                <RotateCw className="h-4 w-4" />
            </Button>

            <Button variant="outline" size="icon" onClick={resetView}>
                <RefreshCw className="h-4 w-4" />
            </Button>

            <Button variant="outline" size="icon" onClick={toggleFullscreen}>
                {fullscreen ? <Minimize className="h-4 w-4" /> : <Expand className="h-4 w-4" />}
            </Button>
        </div>
    )

    return (
        <div
            ref={containerRef}
            className={cn(
                "relative flex flex-col items-center",
                fullscreen ? "fixed inset-0 z-50 bg-background p-4" : "h-full w-full",
            )}
        >
            <div
                className={cn(
                    "relative overflow-hidden rounded-md bg-muted/30",
                    fullscreen ? "flex-1 w-full" : "h-full w-full",
                )}
            >
                <canvas
                    ref={canvasRef}
                    className="cursor-grab max-w-full max-h-full object-contain"
                    style={{
                        transform: `translate(${position.x}px, ${position.y}px)`,
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                />
            </div>

            <div className={cn("flex items-center mt-2 w-full justify-center", fullscreen ? "absolute bottom-4" : "")}>
                {isMobile || window.innerWidth < 640 ? renderCompactControls() : renderFullControls()}
            </div>
        </div>
    )
}
