# Download STB headers
function(download_stb_headers)
    set(STB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stb")
    
    if(NOT EXISTS "${STB_DIR}")
        message(STATUS "Downloading STB headers...")
        file(MAKE_DIRECTORY "${STB_DIR}")
        
        # Download stb_image.h
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
            "${STB_DIR}/stb_image.h"
            STATUS STB_IMAGE_STATUS
        )
        list(GET STB_IMAGE_STATUS 0 STB_IMAGE_RESULT)
        if(NOT STB_IMAGE_RESULT EQUAL 0)
            message(WARNING "Failed to download stb_image.h")
        else()
            message(STATUS "Successfully downloaded stb_image.h")
        endif()
        
        # Download stb_image_write.h
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
            "${STB_DIR}/stb_image_write.h"
            STATUS STB_IMAGE_WRITE_STATUS
        )
        list(GET STB_IMAGE_WRITE_STATUS 0 STB_IMAGE_WRITE_RESULT)
        if(NOT STB_IMAGE_WRITE_RESULT EQUAL 0)
            message(WARNING "Failed to download stb_image_write.h")
        else()
            message(STATUS "Successfully downloaded stb_image_write.h")
        endif()
        
        # Download stb.h (common header)
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/nothings/stb/master/stb.h"
            "${STB_DIR}/stb.h"
            STATUS STB_HEADER_STATUS
        )
        list(GET STB_HEADER_STATUS 0 STB_HEADER_RESULT)
        if(NOT STB_HEADER_RESULT EQUAL 0)
            message(WARNING "Failed to download stb.h")
        else()
            message(STATUS "Successfully downloaded stb.h")
        endif()
    else()
        message(STATUS "STB headers already exist at ${STB_DIR}")
    endif()
endfunction()
